# GPU library benchmarks

| Script | Purpose |
| --- | --- |
| `hipblaslt.py` | Run hipBLASLt benchmark configurations and collect results |
| `miopen.py` | Run MIOpen driver configurations and collect results |
| `bmm_to_hipblaslt.py` | Convert batched-matmul arguments to a hipBLASLt benchmark command |
| `gen_miopen_config.py` | Convert convolution names to MIOpen benchmark configurations |

These tools belong together because they prepare or run GPU library benchmarks. Run each script's usage or help for its inputs. Benchmark execution requires the corresponding GPU libraries on the execution host; the command converters do not run GPU workloads.

Example conversion:

```bash
python3 bench/gen_miopen_config.py conv_2d_bfloat16_forward_16x192x128x32_nhwc_40x1x1x32_fhwc_nhwf_1x1s_0x0p_1x1d_1g
```
