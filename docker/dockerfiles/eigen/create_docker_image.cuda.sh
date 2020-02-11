image_name=rocm/eigen-test:cuda-9.0-gitlab
docker build -t $image_name -f Dockerfile.cuda .
