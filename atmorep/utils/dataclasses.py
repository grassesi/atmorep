from collections.abc import NamedTuple
from numbers import Number
from dataclasses import dataclass
from typing import Literal

from strenum import StrEnum
import numpy as np


class TimeLevLatLon(NamedTuple):
    time: int
    lev: int
    lat: int
    lon: int


class TimeLatLon(NamedTuple):
    time: int
    lat: int
    lon: int

    def as_latlon(self):
        return LatLon(self.lat, self.lon)


class LatLon(NamedTuple):
    lat: float | int
    lon: float | int

    def as_int(self):
        return LatLon(int(self.lat), int(self.lon))

    def __mul__(self, other):
        if isinstance(other, Number):
            return LatLon(self.lat * other, self.lon * other)
        elif isinstance(other, LatLon):
            return LatLon(self.lat * other.lat, self.lon * other.lon)
        else:
            raise TypeError("unsupported type")

    def __div__(self, other):
        if isinstance(other, Number):
            return LatLon(self.lat / other, self.lon / other)
        elif isinstance(other, LatLon):
            return LatLon(self.lat / other.lat, self.lon / other.lon)
        else:
            raise TypeError("unsupported type")
    
    def __mod__(self, other):
        if isinstance(other, Number):
            return LatLon(self.lat % other, self.lon % other)
        elif isinstance(other, LatLon):
            return LatLon(self.lat % other.lat, self.lon % other.lon)
        else:
            raise TypeError("unsupported type")

    def __sub__(self, other):
        if isinstance(other, Number):
            return LatLon(self.lat - other, self.lon - other)
        elif isinstance(other, LatLon):
            return LatLon(self.lat - other.lat, self.lon - other.lon)
        else:
            raise ValueError("unsupported type")

    def __le__(self, other):
        if isinstance(other, LatLon):
            return all(self.lat <= other.lat, self.lon <= other.lon)
        else:
            raise TypeError("unsupported type")


class Dimensions(StrEnum):
    datetime = "datetime"
    lev = "lev"
    lat = "lat"
    lon = "lon"

class SpatialDimensions(StrEnum):
    lev = "lev"
    lat = "lat"
    lon = "lon"

class OutputType(StrEnum):
    source = "source"
    target = "target"
    pred = "pred"
    ens = "ens"

class TokenInfo(NamedTuple):
    year: int
    day: int
    hour: int
    vlvl: int
    lat: int
    lon: int
    vlvl_: int
    res: int


class Field(NamedTuple):
    name: str
    data: np.ndarray

class OutputData(NamedTuple):
    fields: list[Field]
    coords: dict[Dimensions, list[np.ndarray]]

@dataclass
class FileParams:
    shape: LatLon
    coords_min: LatLon
    coords_max: LatLon
    resolution: float
    
    @classmethod
    def from_config(cls, config):
        shape = LatLon(*config.file_shape[1:])
        geo_range = config.geo_range_sampling
        coords_min = LatLon(geo_range[0][0]+90, geo_range[1][0])
        coords_max = LatLon(geo_range[0][1]+90, geo_range[1][1])
        
        resolution = coords_max.lon / shape.lon
        
        return cls(shape, coords_min, coords_max, resolution)
    
@dataclass
class TokenParams:
    num: TimeLatLon
    size: TimeLatLon
    overlap: LatLon

    @classmethod
    def from_field(cls, field_idx: int, config):
        field = config.fields[field_idx]
        n_sample = TimeLatLon(*field[3]),
        size = TimeLatLon(*field[4])
        overlap = LatLon(*config.token_overlap)
        
        return cls(n_sample, size, overlap)
        
    def get_sample_len_overlap(
        self, field_params: FileParams
    ) -> tuple[LatLon]:
        token_len = self.size.as_latlon() * field_params.resolution
        sample_len = self.num.as_latlon() * token_len
        sample_overlap = self.overlap * token_len
        return sample_len, sample_overlap
        
@dataclass
class SamplingParams:
    samples_per_batch: int
    batches_per_date: int


@dataclass
class FieldParams:
    field_name: str
    vlvls: list[int]
    token_params: TokenParams
    normalization: Literal["local", "global"]

    @classmethod
    def from_field(cls, field_idx: int, config):
        field = config.fields[field_idx]
        name = field[0]
        lvls = field[2]
        tokens = TokenParams.from_field(field, config)
        normalization = field[6]
        return cls(name, lvls, tokens, normalization)
    
class Coords:
    def __init__(self, start: float, bound: float, stepsize: float, extra=None):
        self.start = start
        self.bound = bound
        self.stepsize = stepsize
        self._extra = extra

    @property
    def n_tiles(self):
        extra = 0
        if self.has_extra:
            extra = 1
        return self._range_n_tiles + extra

    @property
    def _range_n_tiles(self):
        return int((self.bound - self.start) / self.stepsize)

    def from_index(self, i):
        if i < self.n_tiles:
            if i < self._range_n_tiles:
                return self.start + i * self.stepsize
            else:
                return self._extra
        else:
            raise IndexError

    def _range_value_from_index(self, i):
        return

    def get_sample_mid_points(self):
        return np.array([sel    f.from_index(i) for i in range(self.n_tiles)])
    
    def get_grid_points(self):
        pass

    @property
    def has_extra(self):
        return not self._extra is None