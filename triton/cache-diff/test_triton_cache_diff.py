import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("tcd", Path(__file__).with_name("triton_cache_diff.py"))
tcd = importlib.util.module_from_spec(spec); spec.loader.exec_module(tcd)


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)


def test_verdicts(tmp_path):
    b, a = tmp_path / "b", tmp_path / "a"
    _write(b, "h1/K.hsaco", "x"); _write(a, "h9/K.hsaco", "x")
    _write(b, "h1/K.amdgcn", "\t.loc 1 10 0\n\tv_add v0, v1, v2 ; c\n"); _write(a, "h9/K.amdgcn", "\t.loc 1 12 0\n\tv_add v0, v1, v2\n")
    _write(b, "h2/L.hsaco", "y"); _write(a, "h8/L.hsaco", "z")
    _write(a, "h7/M.hsaco", "m")
    rows = {(r["kernel"], r["ext"]): r["verdict"] for r in tcd.compare(tcd.scan(b, {"hsaco", "amdgcn"}), tcd.scan(a, {"hsaco", "amdgcn"}))}
    assert rows == {("K", "hsaco"): "identical", ("K", "amdgcn"): "same-code", ("L", "hsaco"): "differs", ("M", "hsaco"): "only-after"}
    assert tcd.main([str(b), str(a)]) == 1


def test_elf_sections_ignore_debug():
    import struct
    def elf(text: bytes, debug: bytes) -> bytes:
        names = b"\0.text\0.debug_line\0.shstrtab\0"
        body = text + debug + names
        hdr = bytearray(64); hdr[:5] = b"\x7fELF\x02"
        shoff = 64 + len(body)
        struct.pack_into("<Q", hdr, 0x28, shoff); struct.pack_into("<HHH", hdr, 0x3A, 64, 4, 3)
        def sec(name, off, size): return struct.pack("<IIQQQQIIQQ", name, 1, 0, 0, off, size, 0, 0, 1, 0)
        shs = sec(0, 0, 0) + sec(1, 64, len(text)) + sec(7, 64 + len(text), len(debug)) + sec(19, 64 + len(text) + len(debug), len(names))
        return bytes(hdr) + body + shs
    assert tcd.elf_code_sections(elf(b"CODE", b"/a/x.py")) == tcd.elf_code_sections(elf(b"CODE", b"/bb/y.py"))
    assert tcd.elf_code_sections(elf(b"CODE", b"")) != tcd.elf_code_sections(elf(b"CODX", b""))
