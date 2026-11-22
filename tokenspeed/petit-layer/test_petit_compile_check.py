import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("petit_compile_check", HERE / "petit_compile_check.py")
pcc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcc)

ASM_A = """\t.text
\t.file\t1 "/a/tree/mega_moe" "k.py"
k:
\t.loc\t1 695 0 prologue_end ; k.py:695:0
\ts_load_dwordx2 s[0:1], s[0:1], 0x0 ; comment
\tv_add_f32_e32 v1, v2, v3
.Lfunc_end0:
\t.size\tk, .Lfunc_end0-k
"""
ASM_B = ASM_A.replace("/a/tree", "/another/tree").replace("695", "723")


def test_instructions_drop_directives_paths_and_comments():
    assert pcc.instructions(ASM_A) == ["s_load_dwordx2 s[0:1], s[0:1], 0x0", "v_add_f32_e32 v1, v2, v3"]


def test_compare_ignores_paths_and_lines_but_not_code():
    assert pcc.compare(ASM_A, ASM_B)["identical"]
    assert not pcc.compare(ASM_A, ASM_B.replace("v_add_f32_e32", "v_mul_f32_e32"))["identical"]


def test_cli_compare_exit_code(tmp_path):
    a, b = tmp_path / "a.s", tmp_path / "b.s"
    a.write_text(ASM_A)
    b.write_text(ASM_B)
    assert pcc.main(["compare", str(a), str(b)]) == 0
    b.write_text(ASM_B + "\ts_endpgm\n")
    assert pcc.main(["compare", str(a), str(b)]) == 1


def test_usage_describes_every_kernel_choice():
    table = [line.split("  ")[1] for line in pcc.__doc__.splitlines() if line.startswith("  ") and line[2] != " "]
    named = {k.strip(" ,") for row in table for k in row.split(", ")}
    assert set(pcc.KERNELS) <= named, set(pcc.KERNELS) - named


def test_compile_names_the_tokenspeed_it_needs():
    class UpstreamTest:  # the owner test module of a TokenSpeed without _megamoe_config
        __file__ = "test_gluon_petit.py"
    with pytest.raises(SystemExit, match="46671ec2c"):
        pcc.require_megamoe_config(UpstreamTest)
    UpstreamTest._megamoe_config = object()
    pcc.require_megamoe_config(UpstreamTest)
