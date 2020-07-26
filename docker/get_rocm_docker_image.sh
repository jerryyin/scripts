# docker_repo=rocm/eigen-test
# tag=rocm-3.3

# docker_repo=rocm/tensorflow
# tag=rocm3.5-tf2.2-ofed4.6-openmpi4.0.0-horovod
# tag=rocm3.3-tf1.15-ofed4.6-openmpi4.0.0-horovod
# tag=rocm3.3-tf2.1-ofed4.6-openmpi4.0.0-horovod
# tag=rocm3.1-tf1.15-ofed4.6-openmpi4.0.0-horovod
# tag=rocm3.3-tf2.1-dev
# tag=rocm3.3-tf1.15-dev
# tag=rocm3.1-tf2.1-python3
# tag=rocm2.9-tf1.15-dev

# docker_repo=rocm/tensorflow-autobuilds
# tag=rocm3.5-b6f54db # 200629
# tag=rocm3.3-9ca344d # 200618
# tag=rocm3.3-csb-897803b # 200630

# docker_repo=rocm/tensorflow-private
# tag=rocm3.6-rc3-tf2.1-swdev241977-vdi-from-src
# tag=rocm3.6-rc3-tf2.1-swdev241977
# tag=tf-develop-upstream-ofed4.6-openmpi4.0.0-horovod-debug
# tag=rocm3.5-tf2.2-enhanced-amp-dev-horovod
# tag=rocm3.3-tf2.2-enhanced-ofed4.6-openmpi4.0.0-horovod-amp

docker_repo=devenamd/tensorflow
tag=rocm36-tf-rocmfork-200713

# docker_repo=devenamd/tensorflow
# tag=rocm3.3-200408
# tag=rocm3.0_191220

# docker_repo=devenamd/mlir
# tag=rocm-3.1-200309

# docker_repo=sunway513/hiptensorflow
# tag=

# docker_repo=tensorflow/tensorflow
# tag=nightly-py3

# docker_repo=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang
# tag=2890_ubuntu_py3_tensorflow_r2.1-hipclang
# tag=2226_ubuntu_py3_tensorflow_master-hipclang

docker_image=$docker_repo:$tag
container_name=deven_06_rocm36_tf_rocmfork

options=""
options="$options -it"
options="$options --network=host"
options="$options --ipc=host"
options="$options --shm-size 16G"
options="$options --group-add video"
options="$options --cap-add=SYS_PTRACE"
options="$options --security-opt seccomp=unconfined"

options="$options --device=/dev/kfd"
options="$options --device=/dev/dri"

options="$options -v $HOME/deven/common:/common"
# options="$options -v /data-bert:/data-bert"
# options="$options -v /data/imagenet-inception:/imagenet"
# options="$options -v /data/imagenet-inception:/data/imagenet-inception:"

docker run $options --name $container_name $docker_image
