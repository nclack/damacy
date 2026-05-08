# Run doctests embedded in damacy module docstrings. Examples that touch
# CUDA / a real zarr store stay in ``::`` literal blocks (doctest skips
# them); only pure-Python examples (Dtype.coerce, Sample, Config) carry
# ``>>>`` prompts and are exercised here.
from __future__ import annotations

import doctest

import damacy


def test_damacy_doctests() -> None:
    results = doctest.testmod(
        damacy,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} doctest(s) failed in damacy"
    )
    # Sanity: at least one example actually ran. Catches a broken finder
    # before it silently green-lights an empty pass.
    assert results.attempted > 0
