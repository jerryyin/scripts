docker_repo=devenamd/ubuntu
tag=develop.16.04

docker_image=$docker_repo:$tag

container_name=deven_MLIR

docker run \
       -it \
       --network=host \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add video \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/deven/common:/common \
       --name $container_name \
       $docker_image \
