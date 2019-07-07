#docker_image=rocm/rocm-terminal:1.9.0
#docker_image=rocm/tensorflow:rocm1.8.2-tf1.8-python3-dev
#docker_image=devenamd/eigen:eigen-upstream-rocm-1.9.0
#docker_image=rocmqa/tf18-9083-gfx900:ub18.04
#docker_image=rocm/tensorflow:rocm1.9.0-tf1.10-python3-dev
#docker_image=sunway513/hiptensorflow:rocm1.9.1-tf1.10-resnet50-fp16-v9
#docker_image=sunway513/hiptensorflow:rocm1.9.1-tf1.12-dev-v2
#docker_image=sunway513/hiptensorflow:rocm1.9.2-tf1.10-xla
#docker_image=sunway513/hiptensorflow:rocm2.0-tf1.13-rc0-dev-v2
#docker_image=sunway513/hiptensorflow:rocm2.1-tf1.12-python3-dev-2
#docker_image=rocm/tensorflow:rocm2.1-tf1.12-python3
#docker_image=whchung/hiptensorflow:rocm2.1-tf1.12-python3-dynamic-load-dev

#docker_image=rocm/tensorflow:rocm2.1-tf1.12-python3-dev
#docker_image=rocm/eigen-test:rocm-2.4

#docker_image=devenamd/tensorflow:rocm24_190509
docker_image=rocm/tensorflow-autobuilds:dev-latest

container_name=deven_rocm24_tf_miopen_immediate_mode_2

docker run \
       -it \
       --network=host \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add video \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/deven/common:/common \
       -v $HOME/deven/dockerx:/dockerx \
       --name $container_name \
       $docker_image
