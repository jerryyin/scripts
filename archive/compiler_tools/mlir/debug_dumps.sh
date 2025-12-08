if [ $1 == "-on" ]; then
    
    export MLIR_ROCM_DUMP_AFTER_CONV_GPU_KERNEL=1
    export MLIR_ROCM_DUMP_AFTER_GEN_HSACO_GETTER=1
    export MLIR_ROCM_DUMP_AFTER_CONV_GPU_LAUNCH_FUNC=1

    export HIP_TRACE_API=1
    export HIP_DB=api+mem+cpy

    export HCC_DB=0x48a
    
    # : is the no-op command for bash
    :
    
elif [ $1 == "-off" ]; then
    
    export -n MLIR_ROCM_DUMP_AFTER_CONV_GPU_KERNEL=1
    export -n MLIR_ROCM_DUMP_AFTER_GEN_HSACO_GETTER=1
    export -n MLIR_ROCM_DUMP_AFTER_CONV_GPU_LAUNCH_FUNC=1

    export -n HIP_TRACE_API=1
    export -n HIP_DB=api+mem+cpy

    export -n HCC_DB=0x48a
    
fi

