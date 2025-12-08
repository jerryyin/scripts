set -e

WORKDIR=$1
ROCM_PATH=$2

cd $WORKDIR

OPENMPI_VERSION="4.0"
OPENMPI_PATCHLEVEL="0"
PKG_NAME=openmpi-${OPENMPI_VERSION}.${OPENMPI_PATCHLEVEL}

wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION}/${PKG_NAME}.tar.gz
tar -zxf ${PKG_NAME}.tar.gz
cd ${PKG_NAME}
./configure --prefix=${ROCM_PATH}/openmpi --enable-orterun-prefix-by-default
make -j $(nproc) && make install

OPENMPI_HOME="${ROCM_PATH}/openmpi"

# Create a wrapper for OpenMPI to allow running as root by default
mv ${OPENMPI_HOME}/bin/mpirun ${OPENMPI_HOME}/bin/mpirun.real && \
    echo '#!/bin/bash' > ${OPENMPI_HOME}/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> ${OPENMPI_HOME}/bin/mpirun && \
    chmod a+x ${OPENMPI_HOME}/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
echo "hwloc_base_binding_policy = none" >> ${ROCM_PATH}/openmpi/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> ${ROCM_PATH}/openmpi/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0"  >> ${ROCM_PATH}/openmpi/etc/openmpi-mca-params.conf

# export PATH="${OPENMPI_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:${LD_LIBRARY_PATH}"
