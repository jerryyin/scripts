image_name=devenamd/tfrt:cpu_200517

docker build -t $image_name -f Dockerfile .  && \
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
       $image_name
