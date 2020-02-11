# docker_repo=rocm/eigen-test
# tag=rocm-3.0

# docker_repo=rocm/tensorflow
# tag=rocm2.9-tf1.15-dev

docker_repo=rocm/tensorflow-autobuilds
# tag=rocm3.0-csb-1f404dc
tag=rocm3.0-fba6e87
# tag=rocm3.0-26c8e05

# docker_repo=rocm/tensorflow-private
# tag=rocm2.10-tf-rocsparse-test
# tag=rocm2.9-tf2.0-roctracer-v5
# tag=rocm2.9-tf1.15-roctracer-v3
# tag=rocm2.9-tf1.15-roctracer-v2
# tag=rocm2.9-tf1.15-roctracer-v1

# docker_repo=devenamd/tensorflow
# tag=fused_batchnorm_bug

 
# docker_repo=devenamd/tensorflow
# tag=rocm3.0_191220

# docker_repo=devenamd/mlir
# tag=rocm-2.6-latest

# docker_repo=sunway513/hiptensorflow
#

docker_image=$docker_repo:$tag

container_name=deven_rocm30_tf_rocmfork_weekly_sync

options=""
options="$options -it"
options="$options --network=host"
options="$options --ipc=host"
options="$options --shm-size 16G"
options="$options --device=/dev/kfd"
options="$options --device=/dev/dri"
options="$options --group-add video"
options="$options --cap-add=SYS_PTRACE"
options="$options --security-opt seccomp=unconfined"
options="$options -v $HOME/deven/common:/common"
# options="$options -v /data/imagenet-inception:/imagenet"

docker run $options --name $container_name $docker_image
