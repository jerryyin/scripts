; 8 contiguous UNIFORM loads from a noalias readonly kernel-arg ptr, packed into a
; descriptor and consumed. Tests whether i16 (sub-dword) vs i32 (dword) uniform
; contiguous loads coalesce to WIDE SMEM (s_load_b*) or fall to VMEM (global_load).
define amdgpu_kernel void @t16(ptr addrspace(1) noalias readonly %p, ptr addrspace(1) %o) {
  %a0=getelementptr i16,ptr addrspace(1) %p,i64 0  %v0=load i16,ptr addrspace(1) %a0,!invariant.load !0
  %a1=getelementptr i16,ptr addrspace(1) %p,i64 1  %v1=load i16,ptr addrspace(1) %a1,!invariant.load !0
  %a2=getelementptr i16,ptr addrspace(1) %p,i64 2  %v2=load i16,ptr addrspace(1) %a2,!invariant.load !0
  %a3=getelementptr i16,ptr addrspace(1) %p,i64 3  %v3=load i16,ptr addrspace(1) %a3,!invariant.load !0
  %a4=getelementptr i16,ptr addrspace(1) %p,i64 4  %v4=load i16,ptr addrspace(1) %a4,!invariant.load !0
  %a5=getelementptr i16,ptr addrspace(1) %p,i64 5  %v5=load i16,ptr addrspace(1) %a5,!invariant.load !0
  %a6=getelementptr i16,ptr addrspace(1) %p,i64 6  %v6=load i16,ptr addrspace(1) %a6,!invariant.load !0
  %a7=getelementptr i16,ptr addrspace(1) %p,i64 7  %v7=load i16,ptr addrspace(1) %a7,!invariant.load !0
  %s=add i16 %v0,%v1  store i16 %s, ptr addrspace(1) %o
  store i16 %v2, ptr addrspace(1) %o
  store i16 %v7, ptr addrspace(1) %o
  ret void
}
define amdgpu_kernel void @t32(ptr addrspace(1) noalias readonly %p, ptr addrspace(1) %o) {
  %a0=getelementptr i32,ptr addrspace(1) %p,i64 0  %v0=load i32,ptr addrspace(1) %a0,!invariant.load !0
  %a1=getelementptr i32,ptr addrspace(1) %p,i64 1  %v1=load i32,ptr addrspace(1) %a1,!invariant.load !0
  %a2=getelementptr i32,ptr addrspace(1) %p,i64 2  %v2=load i32,ptr addrspace(1) %a2,!invariant.load !0
  %a3=getelementptr i32,ptr addrspace(1) %p,i64 3  %v3=load i32,ptr addrspace(1) %a3,!invariant.load !0
  %s=add i32 %v0,%v1 store i32 %s, ptr addrspace(1) %o
  store i32 %v2, ptr addrspace(1) %o
  store i32 %v3, ptr addrspace(1) %o
  ret void
}
!0 = !{}
