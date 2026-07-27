import pytest

from uv_torch_compass.index_url import canonical_official_pytorch_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://download.pytorch.org/whl/cu128/",
            "https://download.pytorch.org/whl/cu128",
        ),
        (
            "https://download.pytorch.org:443/whl/nightly/cpu",
            "https://download.pytorch.org/whl/nightly/cpu",
        ),
        ("http://download.pytorch.org/whl/cu128", None),
        ("https://download.pytorch.org.evil.invalid/whl/cu128", None),
        ("https://user@download.pytorch.org/whl/cu128", None),
        ("https://download.pytorch.org/whl/cu128?token=secret", None),
        ("https://download.pytorch.org/whl/rocm6.2", None),
    ],
)
def test_canonicalizes_only_exact_official_indexes(
    value: str,
    expected: str | None,
) -> None:
    assert canonical_official_pytorch_url(value) == expected
