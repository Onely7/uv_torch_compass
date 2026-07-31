"""Read ELF dynamic dependencies without loading third-party native code."""

from __future__ import annotations

import stat
import struct
from pathlib import Path

from uv_torch_compass.errors import ProbeError

_ELF_MAGIC = b"\x7fELF"
_SHT_DYNAMIC = 6
_DT_NULL = 0
_DT_NEEDED = 1
_MAX_SECTION_COUNT = 8_192
_MAX_SECTION_TABLE_BYTES = 4 * 1024 * 1024
_MAX_DYNAMIC_BYTES = 16 * 1024 * 1024
_MAX_STRING_TABLE_BYTES = 16 * 1024 * 1024
_MAX_LIBRARY_NAME_BYTES = 1_024


def read_elf_needed(path: Path, *, root: Path | None = None) -> tuple[str, ...]:
    """Return shared-library names declared by an ELF file.

    Args:
        path: Native library inspected without executing it.
        root: Optional directory that must contain the resolved file.

    Raises:
        ProbeError: If the path is unsafe or the ELF structure is malformed.
    """
    resolved = _validate_path(path, root)
    try:
        with resolved.open("rb") as stream:
            identity = _read_exact(stream, 16, context="ELF identity")
            if identity[:4] != _ELF_MAGIC:
                raise ProbeError(f"native library is not ELF: {path.name}")
            elf_class = identity[4]
            data_encoding = identity[5]
            endian = "<" if data_encoding == 1 else ">" if data_encoding == 2 else ""
            if not endian:
                raise ProbeError("ELF file uses an unsupported byte order")
            if elf_class == 1:
                header_format = f"{endian}HHIIIIIHHHHHH"
                section_format = f"{endian}IIIIIIIIII"
                dynamic_format = f"{endian}II"
            elif elf_class == 2:
                header_format = f"{endian}HHIQQQIHHHHHH"
                section_format = f"{endian}IIQQQQIIQQ"
                dynamic_format = f"{endian}QQ"
            else:
                raise ProbeError("ELF file uses an unsupported word size")

            header = struct.unpack(
                header_format,
                _read_exact(
                    stream,
                    struct.calcsize(header_format),
                    context="ELF header",
                ),
            )
            section_offset = header[5]
            section_entry_size = header[10]
            section_count = header[11]
            sections = _read_sections(
                stream,
                file_size=resolved.stat().st_size,
                offset=section_offset,
                entry_size=section_entry_size,
                count=section_count,
                section_format=section_format,
            )
            libraries: list[str] = []
            for section in sections:
                if section[1] != _SHT_DYNAMIC:
                    continue
                libraries.extend(
                    _read_dynamic_libraries(
                        stream,
                        file_size=resolved.stat().st_size,
                        section=section,
                        sections=sections,
                        dynamic_format=dynamic_format,
                    )
                )
            return tuple(dict.fromkeys(libraries))
    except OSError as exc:
        raise ProbeError(
            f"failed to inspect native library {path.name}: {exc}"
        ) from exc


def _validate_path(path: Path, root: Path | None) -> Path:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ProbeError("native library is not a regular file")
        resolved = path.resolve(strict=True)
        if root is not None and not resolved.is_relative_to(root.resolve(strict=True)):
            raise ProbeError("native library escaped the candidate environment")
        return resolved
    except OSError as exc:
        raise ProbeError(f"failed to inspect native library path: {exc}") from exc


def _read_sections(
    stream,
    *,
    file_size: int,
    offset: int,
    entry_size: int,
    count: int,
    section_format: str,
) -> tuple[tuple[int, ...], ...]:
    expected_size = struct.calcsize(section_format)
    if count <= 0 or count > _MAX_SECTION_COUNT:
        raise ProbeError("ELF file has an unsupported section count")
    if entry_size < expected_size or entry_size > 1_024:
        raise ProbeError("ELF file has an invalid section entry size")
    table_size = entry_size * count
    if table_size > _MAX_SECTION_TABLE_BYTES:
        raise ProbeError("ELF section table exceeds the safety limit")
    _require_file_range(offset, table_size, file_size, context="ELF section table")
    sections: list[tuple[int, ...]] = []
    for position in range(count):
        stream.seek(offset + position * entry_size)
        sections.append(
            struct.unpack(
                section_format,
                _read_exact(stream, expected_size, context="ELF section entry"),
            )
        )
    return tuple(sections)


def _read_dynamic_libraries(
    stream,
    *,
    file_size: int,
    section: tuple[int, ...],
    sections: tuple[tuple[int, ...], ...],
    dynamic_format: str,
) -> tuple[str, ...]:
    dynamic_offset = section[4]
    dynamic_size = section[5]
    string_table_index = section[6]
    if dynamic_size > _MAX_DYNAMIC_BYTES:
        raise ProbeError("ELF dynamic section exceeds the safety limit")
    _require_file_range(
        dynamic_offset,
        dynamic_size,
        file_size,
        context="ELF dynamic section",
    )
    if string_table_index >= len(sections):
        raise ProbeError("ELF dynamic section references an invalid string table")
    string_section = sections[string_table_index]
    string_offset = string_section[4]
    string_size = string_section[5]
    if string_size > _MAX_STRING_TABLE_BYTES:
        raise ProbeError("ELF string table exceeds the safety limit")
    _require_file_range(
        string_offset,
        string_size,
        file_size,
        context="ELF dynamic string table",
    )
    stream.seek(string_offset)
    strings = _read_exact(stream, string_size, context="ELF dynamic string table")

    entry_size = struct.calcsize(dynamic_format)
    if dynamic_size % entry_size:
        raise ProbeError("ELF dynamic section has a truncated entry")
    libraries: list[str] = []
    for position in range(0, dynamic_size, entry_size):
        stream.seek(dynamic_offset + position)
        tag, value = struct.unpack(
            dynamic_format,
            _read_exact(stream, entry_size, context="ELF dynamic entry"),
        )
        if tag == _DT_NULL:
            break
        if tag != _DT_NEEDED:
            continue
        libraries.append(_read_string(strings, value))
    return tuple(libraries)


def _read_string(table: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(table):
        raise ProbeError("ELF dependency name has an invalid string offset")
    terminator = table.find(b"\0", offset, offset + _MAX_LIBRARY_NAME_BYTES)
    if terminator < 0:
        raise ProbeError("ELF dependency name is not safely terminated")
    try:
        value = table[offset:terminator].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError("ELF dependency name is not valid UTF-8") from exc
    if not value:
        raise ProbeError("ELF dependency name must not be empty")
    return value


def _require_file_range(
    offset: int,
    size: int,
    file_size: int,
    *,
    context: str,
) -> None:
    if offset < 0 or size < 0 or offset > file_size or size > file_size - offset:
        raise ProbeError(f"{context} points outside the file")


def _read_exact(stream, size: int, *, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ProbeError(f"{context} is truncated")
    return value
