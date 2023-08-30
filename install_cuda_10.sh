wget http://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
chmod +x ./cuda_10.0.130_410.48_linux
./cuda_10.0.130_410.48_linux --silent --toolkit
rm ./cuda_10.0.130_410.48_linux
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.0/include/
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/
sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CPATH=/usr/local/cuda-10.0/include:$CPATH' >> ~/.bashrc
git clone https://github.com/charlesq34/pointnet2 ~/pointnet2
wget https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
pip2 install ./tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
rm ./tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
