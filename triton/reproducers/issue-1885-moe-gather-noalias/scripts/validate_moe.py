import sys
import moe_gfx1250 as m

# args: schedule mnk_opt(or 'None')
sched = sys.argv[1]
mnk = None if sys.argv[2] == "None" else sys.argv[2]
# block matches benchmark: BM=128, BN=256, BK=256; aligned shape; gather; fused swiglu
try:
    m.test_matmul(512, 512, 512, 128, 256, 256, "float8_e4m3fn", "mxfloat4_e2m1",
                  True, False, False, False, (1.1, 1.4), 2, sched, False, 4, mnk)
    print(f"RESULT PASS schedule={sched} mnk_opt={mnk}")
except Exception as e:
    print(f"RESULT SKIP_OR_FAIL schedule={sched} mnk_opt={mnk}: {type(e).__name__}: {e}")
