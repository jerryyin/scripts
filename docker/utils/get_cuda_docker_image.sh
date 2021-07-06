
docker_repo=tensorflow/tensorflow
# docker_repo=devenamd/tensorflow

# tag=1.12.0-devel-gpu-py3
# tag=cuda_190513
tag=latest-devel-gpu-py3

docker_repo=nvcr.io/nvidia/tensorflow
tag=20.03-tf2-py3

container_name=deven_tf_trial

sudo docker run \
     -it \
     -p 8888:8888 \
     -v $HOME/deven/common:/common \
     --name $container_name \
     $docker_repo:$tag
