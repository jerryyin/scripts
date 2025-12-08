
# docker_repo=tensorflow/tensorflow
# docker_repo=devenamd/tensorflow

# tag=1.12.0-devel-gpu-py3
# tag=cuda_190513
# tag=latest-devel-gpu-py3

docker_repo=nvcr.io/nvidia/tensorflow
tag=21.08-tf2-py3

container_name=deven_01_TF_RESNET50

options=""
options="$options -it"
options="$options --gpus \"device=0,1,2,3,4,5,6,7\""
options="$options --network=host"
options="$options --ipc=host"
options="$options --shm-size 16G"
options="$options --cap-add=SYS_PTRACE"
options="$options --security-opt seccomp=unconfined"

options="$options -v $HOME/deven/common:/common"
options="$options -v /data-bert:/data-bert"
options="$options -v /data/imagenet:/data/imagenet"
# options="$options -v /data/imagenet-inception:/data/imagenet-inception:"

sudo docker run $options --name $container_name $docker_repo:$tag
