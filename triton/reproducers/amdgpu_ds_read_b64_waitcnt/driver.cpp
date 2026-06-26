// Standalone HIP driver for the AMDGPU ds_read_b64 / s_waitcnt miscompile.
// NO Triton dependency. Loads a precompiled HSACO of the _attn_fwd kernel and
// the captured launch descriptor (capture/launch_capture.bin), launches it with
// fixed random inputs, and prints a checksum of the primary output buffer per
// run. A correct build yields identical checksums across runs AND matches the
// -O0 reference; the miscompiled build yields varying checksums (race).
//
// Build:  hipcc -O2 driver.cpp -o driver
// Usage:  ./driver <kernel.hsaco> <nruns>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CHK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %d (%s) at %s:%d\n",e,hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

static const size_t BUF_BYTES = 128ull*1024*1024;   // per I/O buffer (generous)
static const int NPTR_IO = 5;                        // args 0..4 are q,k,v,out,out2

int main(int argc, char** argv){
  if(argc<2){ fprintf(stderr,"usage: %s <hsaco> [nruns]\n",argv[0]); return 1; }
  const char* hsaco_path = argv[1];
  int nruns = (argc>=3)? atoi(argv[2]) : 5;

  // --- read capture descriptor ---
  FILE* fp=fopen("capture/launch_capture.bin","rb");
  if(!fp){ fprintf(stderr,"cannot open capture/launch_capture.bin\n"); return 1; }
  uint32_t hdr[8]; if(fread(hdr,4,8,fp)!=8){fprintf(stderr,"bad hdr\n");return 1;}
  uint32_t gx=hdr[1],gy=hdr[2],gz=hdr[3],bx=hdr[4],by=hdr[5],bz=hdr[6],shmem=hdr[7];
  uint32_t n; if(fread(&n,4,1,fp)!=1){return 1;}
  std::vector<uint32_t> type(n); std::vector<uint64_t> raw(n);
  for(uint32_t i=0;i<n;i++){ uint32_t t; unsigned char s[8];
    if(fread(&t,4,1,fp)!=1||fread(s,8,1,fp)!=1){return 1;}
    type[i]=t; memcpy(&raw[i],s,8); }
  fclose(fp);
  printf("launch: grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%u nargs=%u\n",gx,gy,gz,bx,by,bz,shmem,n);

  // --- allocate I/O device buffers ---
  void* dbuf[NPTR_IO];
  for(int i=0;i<NPTR_IO;i++) CHK(hipMalloc(&dbuf[i], BUF_BYTES));
  // deterministic pseudo-random host fill for inputs (q,k,v = args 0,1,2)
  std::vector<uint16_t> host(BUF_BYTES/2);
  uint32_t seed=12345;
  for(size_t j=0;j<host.size();j++){ seed=seed*1664525u+1013904223u;
    float f=((float)((seed>>8)&0xFFFF)/65535.0f-0.5f)*2.0f; // ~[-1,1]
    uint32_t fb; memcpy(&fb,&f,4); host[j]=(uint16_t)(fb>>16); } // bf16 = high 16 bits of f32
  for(int i=0;i<3;i++) CHK(hipMemcpy(dbuf[i], host.data(), BUF_BYTES, hipMemcpyHostToDevice));

  // --- load module / function ---
  std::vector<char> code; { FILE* f=fopen(hsaco_path,"rb"); if(!f){fprintf(stderr,"no hsaco %s\n",hsaco_path);return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); code.resize(sz); fread(code.data(),1,sz,f); fclose(f); }
  hipModule_t mod; CHK(hipModuleLoadData(&mod, code.data()));
  // find the single kernel symbol
  hipFunction_t fn=nullptr;
  // kernel name is long; read it from a sidecar file produced by reproduce.sh
  char name[512]={0}; { FILE* nf=fopen("asm/kernel_name.txt","r"); if(nf){ fgets(name,sizeof(name),nf); fclose(nf);
      size_t L=strlen(name); if(L&&name[L-1]=='\n') name[L-1]=0; } }
  CHK(hipModuleGetFunction(&fn, mod, name));

  // --- build kernel params (void*[n]) ---
  void* nullp=nullptr;
  std::vector<void*> params(n);
  for(uint32_t i=0;i<n;i++){
    if(i<(uint32_t)NPTR_IO)        params[i]=&dbuf[i];           // q,k,v,out,out2
    else if(i==n-1||i==n-2)        params[i]=&nullp;             // scratch ptrs (NULL)
    else                           params[i]=&raw[i];            // scalars (captured)
  }

  // --- run nruns times, checksum the primary output (arg3) each time ---
  size_t N=BUF_BYTES/2;
  std::vector<uint16_t> out(N), ref(N);
  auto bf16=[&](uint16_t h)->float{ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; };
  double worst=0.0;
  for(int r=0;r<nruns;r++){
    CHK(hipMemset(dbuf[3],0,BUF_BYTES)); CHK(hipMemset(dbuf[4],0,BUF_BYTES));
    CHK(hipModuleLaunchKernel(fn,gx,gy,gz,bx,by,bz,shmem,0,params.data(),nullptr));
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(out.data(), dbuf[3], BUF_BYTES, hipMemcpyDeviceToHost));
    if(r==0){ ref=out; printf("  run 0: reference captured\n"); continue; }
    double md=0.0; long nbad=0;
    for(size_t j=0;j<N;j++){ float a=bf16(out[j]),b=bf16(ref[j]);
      if(std::isfinite(a)&&std::isfinite(b)){ double d=fabs((double)a-b); if(d>md)md=d; if(d>0.01)nbad++; } }
    if(md>worst)worst=md;
    printf("  run %d vs run0: max_abs_diff=%.5f  elems>0.01=%ld\n", r, md, nbad);
  }
  printf("RESULT: worst max_abs_diff across runs = %.5f (tolerance 0.01)\n", worst);
  for(int i=0;i<NPTR_IO;i++) hipFree(dbuf[i]);
  return worst>0.01?2:0;
}
