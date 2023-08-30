
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/6dgraspnet:cuda-18-04
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        grasp_6dof \
        $BASH_OPTION
else
    docker start -i grasp_6dof
fi
