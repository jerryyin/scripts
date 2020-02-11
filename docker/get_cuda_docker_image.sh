
docker_repo=tensorflow/tensorflow
# docker_repo=devenamd/tensorflow

# tag=1.12.0-devel-gpu-py3
# tag=cuda_190513
tag=latest-devel-gpu-py3

container_name=deven_tf_misc_01

sudo nvidia-docker run \
     -it \
     -p 8888:8888 \
     -v $HOME/deven/common:/common \
     --name $container_name \
     $docker_repo:$tag
