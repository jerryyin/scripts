; non-contiguous (scattered) i16 uniform loads -> SMEM or VMEM?
define amdgpu_kernel void @t16nc(ptr addrspace(1) noalias readonly %p, ptr addrspace(1) %o) {
  %a0=getelementptr i16,ptr addrspace(1) %p,i64 0    %v0=load i16,ptr addrspace(1) %a0,!invariant.load !0
  %a1=getelementptr i16,ptr addrspace(1) %p,i64 100  %v1=load i16,ptr addrspace(1) %a1,!invariant.load !0
  %a2=getelementptr i16,ptr addrspace(1) %p,i64 37   %v2=load i16,ptr addrspace(1) %a2,!invariant.load !0
  %a3=getelementptr i16,ptr addrspace(1) %p,i64 5    %v3=load i16,ptr addrspace(1) %a3,!invariant.load !0
  %s0=add i16 %v0,%v1 %s1=add i16 %v2,%v3 %s=add i16 %s0,%s1
  store i16 %s, ptr addrspace(1) %o
  ret void
}
!0 = !{}
