llvm_ir_file=$1
base_name=$(basename $llvm_ir_file .ll)

llvm_bc_file=$base_name.bc
llvm_obj_file=$base_name.o
llvm_hsaco_file=$base_name.hsaco

# # first generate the .bc file from the .ll file using llvm-as
llvm-as $llvm_ir_file

# Then compile the obj/ISA file from the bc file
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -mattr=code-object-v3 $llvm_bc_file

# finally link/build generate the hsaco file from the obj 
ld.lld -flavor gnu -shared $llvm_obj_file -o $llvm_hsaco_file
