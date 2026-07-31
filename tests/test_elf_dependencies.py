import struct
from pathlib import Path

import pytest

from uv_torch_compass.elf_dependencies import read_elf_needed
from uv_torch_compass.errors import ProbeError


def _write_elf(path: Path, libraries: tuple[str, ...]) -> None:
    identity = b"\x7fELF" + bytes((2, 1, 1, 0)) + bytes(8)
    header_size = 64
    section_size = 64
    section_offset = header_size
    strings = bytearray(b"\0")
    string_offsets: list[int] = []
    for library in libraries:
        string_offsets.append(len(strings))
        strings.extend(library.encode("utf-8") + b"\0")
    strings_offset = section_offset + section_size * 3
    dynamic_offset = strings_offset + len(strings)
    dynamic = b"".join(
        struct.pack("<QQ", 1, offset) for offset in string_offsets
    ) + struct.pack("<QQ", 0, 0)
    header = struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        62,
        1,
        0,
        0,
        section_offset,
        0,
        header_size,
        0,
        0,
        section_size,
        3,
        0,
    )
    null_section = bytes(section_size)
    string_section = struct.pack(
        "<IIQQQQIIQQ",
        0,
        3,
        0,
        0,
        strings_offset,
        len(strings),
        0,
        0,
        1,
        0,
    )
    dynamic_section = struct.pack(
        "<IIQQQQIIQQ",
        0,
        6,
        0,
        0,
        dynamic_offset,
        len(dynamic),
        1,
        0,
        8,
        16,
    )
    path.write_bytes(
        identity
        + header
        + null_section
        + string_section
        + dynamic_section
        + strings
        + dynamic
    )


def test_reads_needed_libraries_without_loading_the_elf(tmp_path: Path) -> None:
    library = tmp_path / "vllm/_C.abi3.so"
    library.parent.mkdir()
    _write_elf(library, ("libcudart.so.13", "libtorch.so"))

    assert read_elf_needed(library, root=tmp_path) == (
        "libcudart.so.13",
        "libtorch.so",
    )


def test_rejects_truncated_and_non_elf_files(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.so"
    truncated.write_bytes(b"\x7fELF")
    text = tmp_path / "text.so"
    text.write_bytes(b"not an elf file!!")

    with pytest.raises(ProbeError, match="truncated"):
        read_elf_needed(truncated)
    with pytest.raises(ProbeError, match="not ELF"):
        read_elf_needed(text)


def test_rejects_symlinks_and_paths_outside_the_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.so"
    _write_elf(outside, ("libcudart.so.12",))
    root = tmp_path / "root"
    root.mkdir()
    link = root / "linked.so"
    link.symlink_to(outside)

    with pytest.raises(ProbeError, match="regular file"):
        read_elf_needed(link, root=root)
    with pytest.raises(ProbeError, match="escaped"):
        read_elf_needed(outside, root=root)
