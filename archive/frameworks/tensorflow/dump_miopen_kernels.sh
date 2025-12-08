# You should use llvm-objdump on _temp_0_gfx900.so, not libMIOpen.so.
 
# _temp_0_gfx900.so is generated when running MIOpen, not building it. 
 
# After installing MIOpen, you can run following command to generate _temp_0_gfx900.so
 
# export  AMD_OCL_BUILD_OPTIONS_APPEND="-save-temps-all" /Path/To/MIOpen/bin/MIOpenDriver conv
