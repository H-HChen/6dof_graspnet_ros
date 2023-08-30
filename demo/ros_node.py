#!/usr/bin/env python2


import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rstrip("demo")
sys.path.append(BASE_DIR)

import numpy as np
import grasp_estimator
import tensorflow as tf
import glob
import mayavi.mlab as mlab
from visualization_utils import *
from grasp_data_reader import regularize_pc_point_count
import rospy
from cv_bridge import CvBridge, CvBridgeError
from grasp_msgs.srv import GraspGroup, GraspGroupResponse
from grasp_msgs.srv import CurrentScore, CurrentScoreResponse
from grasp_msgs.msg import GraspPose
from tf import transformations
from sensor_msgs import point_cloud2
import time


class GraspnetNode():
    def __init__(self):
        self.vae_checkpoint_folder = rospy.get_param(
            "~vae_checkpoint_folder",
            os.path.dirname(os.path.abspath(__file__)).rstrip("demo") + "checkpoints/ACRONYM_GAN/")
        self.evaluator_checkpoint_folder = rospy.get_param(
            "~evaluator_checkpoint_folder",
            os.path.dirname(os.path.abspath(__file__)).rstrip("demo") + "checkpoints/ACRONYM_Evaluator/")
        self.gradient_based_refinement = rospy.get_param("~gradient_based_refinement", False)
        self.per_grasp_score = rospy.get_param("~per_grasp_score", False)
        self.visulize = rospy.get_param("~visulize", True)
        self.threshold = rospy.get_param("~threshold", 0.8)
        self.gpu_fraction = rospy.get_param("~gpu_fraction", 1)
        self.cfg = grasp_estimator.joint_config(
            self.vae_checkpoint_folder,
            self.evaluator_checkpoint_folder,
        )
        self.cfg['threshold'] = self.threshold
        self.cfg['sample_based_improvement'] = 1 - int(self.gradient_based_refinement)
        self.cfg['num_refine_steps'] = 10 if self.gradient_based_refinement else 20
        self.cfg['num_samples'] = 1 if self.per_grasp_score else 200
        self.estimator = grasp_estimator.GraspEstimator(self.cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg.gpu)
        if self.gpu_fraction < 1:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        self.estimator.build_network()
        self.estimator.load_weights(self.sess)

        self.bridge = CvBridge()
        self.grasppoint_srv = rospy.Service('nv_graspnet/get_grasp_result', GraspGroup, self.grasp_callback)
        if self.per_grasp_score:
            self.eval_srv = rospy.Service('nv_graspnet/evaluate', CurrentScore, self.eval_grasp)
        self.main_vis = False
        self.tmp_grasp = ()

    def grasp_callback(self, request):
        try:
            image_rgb = self.bridge.imgmsg_to_cv2(request.rgb, "rgb8")
            segmap = self.bridge.imgmsg_to_cv2(request.seg, "passthrough")
            cam_K = np.array(request.K).reshape(3, 3)
            image_depth = self.bridge.imgmsg_to_cv2(request.depth, "passthrough").copy()
            image_depth[np.isnan(image_depth)] = 0.
            segmap_id = request.segmap_id

        except CvBridgeError as e:
            rospy.logerr(e)
        begin = time.time()
        pc_full, pc_segments, pc_colors = backproject(image_depth, image_rgb, cam_K,
                                                      segmap, segmap_id,
                                                      return_finite_depth=True)
        object_pc = pc_segments
        latents = self.estimator.sample_latents()
        generated_grasps, generated_scores, _ = self.estimator.predict_grasps(
            self.sess,
            object_pc,
            latents,
            num_refine_steps=self.cfg.num_refine_steps,
        )

        reponse = GraspGroupResponse()
        for pose_cam, score in zip(generated_grasps, generated_scores):
            grasp_pose = GraspPose()
            grasp_pose.pred_grasps_cam = pose_cam.flatten()
            grasp_pose.score = score
            reponse.grasp_poses.append(grasp_pose)

        # Visualize results
        if self.visulize:
            self.main_vis = True
            self.tmp_grasp = (pc_full, pc_colors, generated_grasps, generated_scores)

        return reponse

    def eval_grasp(self, request):
        pc = point_cloud2.read_points_list(
            request.point_cloud, field_names=("x", "y", "z"))
        pc = np.asarray(pc)
        gripper_pose = np.array(request.pose).reshape([4, 4])
        gripper_trans = gripper_pose[:3, 3]
        gripper_rot = transformations.euler_from_matrix(gripper_pose[:3, :3])
        grasp_score = self.estimator.compute_grasps_score(self.sess, pc, gripper_trans, gripper_rot)
        reponse = CurrentScoreResponse()
        reponse.score = grasp_score

        return reponse


def depth2pc(depth, K, rgb=None):
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x, :]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


def backproject(depth_cv, rgb, intrinsic_matrix, segmap, segmap_id, return_finite_depth=True):
    pc_full, pc_colors = depth2pc(depth_cv, intrinsic_matrix, rgb)

    # Threshold distance
    if return_finite_depth:
        pc_colors = pc_colors[np.isfinite(pc_full[:, 2])]
    pc_full = pc_full[np.isfinite(pc_full[:, 2])]

    # Extract instance point clouds from segmap and depth map
    if segmap is not None:
        obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap > 0])
        if segmap_id not in obj_instances:
            raise RuntimeError("Target ID not in segment map")
        for i in obj_instances:
            if i == segmap_id:
                inst_mask = (segmap == i)
                pc_segment, _ = depth2pc(depth_cv * inst_mask, intrinsic_matrix)
                pc_segment = pc_segment[np.isfinite(pc_segment[:, 2])]

    return pc_full, pc_segment, pc_colors


if __name__ == '__main__':
    rospy.init_node('grasp_pose_estimator', anonymous=True, disable_signals=True)
    rospy.loginfo("Init grasp estimation node")
    graspnet_node = GraspnetNode()
    rate = rospy.Rate(10)
    try:
        while True:
            if graspnet_node.main_vis:
                mlab.figure(bgcolor=(1, 1, 1))
                draw_scene(
                    pc=graspnet_node.tmp_grasp[0],
                    pc_color=graspnet_node.tmp_grasp[1],
                    grasps=graspnet_node.tmp_grasp[2],
                    grasp_scores=graspnet_node.tmp_grasp[3],
                    visualize_diverse_grasps=True
                )
                mlab.show()
                graspnet_node.main_vis = False
                graspnet_node.tmp_grasp = ()
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down...')
