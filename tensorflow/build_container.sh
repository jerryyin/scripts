TF_REPO_LOC=/home/deven/deven/repos/tensorflow-upstream

DOCKERFILE_PATH=$TF_REPO_LOC/tensorflow/tools/ci_build/Dockerfile.rocm
DOCKER_CONTEXT_PATH=$TF_REPO_LOC/tensorflow/tools/ci_build

DOCKER_IMAGE_NAME=devenamd/tensorflow:rocm3.0_200113


cd $TF_REPO_LOC && \
    docker build -t $DOCKER_IMAGE_NAME -f $DOCKERFILE_PATH $DOCKER_CONTEXT_PATH
