from __future__ import annotations

import math
import operator
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from . import BatchSpec, FileMetadataReader, Sample, _component, _native, _positive_int


def _index(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = operator.index(value)
    if not 0 <= result <= (1 << 31) - 1:
        raise ValueError(f"{name} must be between 0 and 2**31 - 1")
    return result


def _finite_number(value: float) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise TypeError("transform and sampler values must be numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("transform and sampler values must be finite")
    return result


@dataclass(frozen=True, slots=True)
class NgffLimits:
    """Maximum levels and total JSON bytes read when loading an image."""

    max_levels: int = 32
    max_metadata_bytes: int = 4 << 20

    def __post_init__(self) -> None:
        _positive_int(self.max_levels, "max_levels", (1 << 31) - 1)
        _positive_int(self.max_metadata_bytes, "max_metadata_bytes", (1 << 64) - 1)


@dataclass(frozen=True, slots=True)
class NgffAxis:
    """An axis in the array's dimension order, with its declared NGFF unit."""

    name: str
    kind: Literal["space", "time", "channel"]
    unit: str | None


@dataclass(frozen=True, slots=True)
class NgffLevel:
    """A source array and its voxel-corner mapping into reference-level indices."""

    uri: str
    shape: tuple[int, ...]
    scale_to_reference: tuple[float, ...]
    origin_reference_index: tuple[float, ...]


@dataclass(frozen=True, slots=True, init=False)
class NgffImage:
    """Load an immutable OME-Zarr 0.5 image description through the given reader.

    Loading reads the image group's metadata and each level's array metadata.
    It finishes before returning and releases the reader for other uses.
    ``multiscale_index`` explicitly selects an entry in ``ome.multiscales``.
    """

    axes: tuple[NgffAxis, ...]
    levels: tuple[NgffLevel, ...]
    data_type: str
    _native: object

    def __init__(
        self,
        uri: str | os.PathLike[str],
        *,
        reader: FileMetadataReader,
        multiscale_index: int,
        limits: NgffLimits | None = None,
    ) -> None:
        if not isinstance(reader, FileMetadataReader):
            raise TypeError("reader must be a FileMetadataReader")
        if limits is None:
            limits = NgffLimits()
        if not isinstance(limits, NgffLimits):
            raise TypeError("limits must be NgffLimits")
        path = os.fspath(uri)
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("uri must be a nonempty path without NUL characters")
        native = _component(
            _native.ngff_load,
            reader._native,
            path,
            _index(multiscale_index, "multiscale_index"),
            limits.max_levels,
            limits.max_metadata_bytes,
        )
        info = _native.ngff_info(native)
        object.__setattr__(self, "_native", native)
        object.__setattr__(
            self, "axes", tuple(NgffAxis(*axis) for axis in info["axes"])
        )
        object.__setattr__(
            self, "levels", tuple(NgffLevel(**level) for level in info["levels"])
        )
        object.__setattr__(self, "data_type", info["data_type"])


@dataclass(frozen=True, slots=True)
class Sampler:
    """Point interpolation and treatment of source indices outside spatial axes.

    Nearest selects ``floor(source_corner)``. Linear interpolates between
    centers at ``index + 0.5``. Constant extends the source with
    ``constant_value``; clamp repeats the closest edge value; error rejects
    a query needing out-of-bounds samples. No extra antialias filter is applied.
    """

    filter: Literal["nearest", "linear"] = "linear"
    boundary: Literal["error", "constant", "clamp"] = "error"
    constant_value: float = 0

    def __post_init__(self) -> None:
        if self.filter not in ("nearest", "linear"):
            raise ValueError("filter must be 'nearest' or 'linear'")
        if self.boundary not in ("error", "constant", "clamp"):
            raise ValueError("boundary must be 'error', 'constant', or 'clamp'")
        value = _finite_number(self.constant_value)
        if self.boundary != "constant" and value != 0:
            raise ValueError("constant_value requires boundary='constant'")
        object.__setattr__(self, "constant_value", value)


@dataclass(frozen=True, slots=True, init=False)
class SpatialQuery:
    """Map a fixed output grid into reference-level voxel-corner coordinates.

    ``output_to_reference`` has rank rows and rank + 1 columns. The last
    column is the translation; the preceding columns are the linear map.
    Rows and columns follow NGFF array axis order, including time/channel
    dimensions, which only allow identity plus integer translation.
    ``level='auto'`` selects the coarsest level no coarser than the output
    spacing in any direction, falling back to level zero for upsampling.
    """

    output_to_reference: tuple[tuple[float, ...], ...]
    sampler: Sampler
    level: int | Literal["auto"]

    def __init__(
        self,
        *,
        output_to_reference: Iterable[Iterable[float]],
        sampler: Sampler,
        level: int | Literal["auto"] = "auto",
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler")
        matrix = tuple(
            tuple(_finite_number(v) for v in row) for row in output_to_reference
        )
        rank = len(matrix)
        if not 2 <= rank <= _native.MAX_RANK or any(
            len(row) != rank + 1 for row in matrix
        ):
            raise ValueError(
                "output_to_reference must have rank rows and rank + 1 columns"
            )
        if isinstance(level, str):
            if level != "auto":
                raise ValueError("level must be 'auto' or a nonnegative integer")
        else:
            level = _index(level, "level")
        object.__setattr__(self, "output_to_reference", matrix)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "level", level)


@dataclass(frozen=True, slots=True, init=False)
class ResolvedSpatialQuery:
    """Owned source selection and geometry, independent of the loaded image.

    Aligned results can be pushed to a Pipeline or converted with ``as_sample``.
    Other results raise UnsupportedOperation when converted or pushed; their
    geometry remains available for inspection without invoking a decoder.
    """

    uri: str
    level: int
    shape: tuple[int, ...]
    source_shape: tuple[int, ...]
    output_to_source: tuple[tuple[float, ...], ...]
    sampler: Sampler
    source_bounds_index: tuple[tuple[int, int], ...]
    read_bounds_index: tuple[tuple[int, int], ...]
    requires_resampling: bool
    _native: object

    @classmethod
    def _from_native(cls, native: object, sampler: Sampler) -> ResolvedSpatialQuery:
        result = object.__new__(cls)
        object.__setattr__(result, "_native", native)
        object.__setattr__(result, "sampler", sampler)
        for name, value in _native.spatial_info(native).items():
            object.__setattr__(result, name, value)
        return result

    def _to_native(self) -> dict[str, Any]:
        return _component(_native.spatial_sample, self._native)

    def as_sample(self) -> Sample:
        """Return an interval sample, or fail if resampling is required."""
        sample = self._to_native()
        return Sample(uri=sample["uri"], aabb=tuple(axis[1] for axis in sample["axes"]))


@dataclass(frozen=True, slots=True)
class SpatialResolver:
    """Resolve queries using injected image metadata and fixed output geometry.

    Resolution does no I/O and can be shared across threads. The same
    BatchSpec should be supplied to the downstream Pipeline.
    """

    image: NgffImage
    output: BatchSpec

    def __post_init__(self) -> None:
        if not isinstance(self.image, NgffImage) or not isinstance(
            self.output, BatchSpec
        ):
            raise TypeError("SpatialResolver requires an NgffImage and a BatchSpec")
        if len(self.image.axes) != len(self.output.shape):
            raise ValueError("output rank must match the NGFF image")

    def resolve(self, query: SpatialQuery) -> ResolvedSpatialQuery:
        """Choose the source level, map coordinates, and bound source reads."""
        if not isinstance(query, SpatialQuery):
            raise TypeError("query must be a SpatialQuery")
        native = _component(
            _native.spatial_resolve,
            self.image._native,
            self.output.shape,
            self.output.samples,
            int(self.output.dtype),
            query.output_to_reference,
            {"nearest": 1, "linear": 2}[query.sampler.filter],
            {"error": 1, "constant": 2, "clamp": 3}[query.sampler.boundary],
            query.sampler.constant_value,
            -1 if query.level == "auto" else query.level,
        )
        return ResolvedSpatialQuery._from_native(native, query.sampler)
