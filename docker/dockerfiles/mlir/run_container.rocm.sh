docker_image=rocm/eigen-test:rocm-2.2
container_name=eigen-upstream-nightly-rocm-`date +%y%m%d`

docker run \
       --rm \
       --network=host \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add video \
       --name $container_name \
       $docker_image \
       /home/rocm-user/run_tests_nightly.rocm.sh
