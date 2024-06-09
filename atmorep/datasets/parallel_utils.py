from datetime import datetime
from collections.abc import NamedTuple
from numbers import Number
from dataclasses import dataclass, 
import itertools as it

import numpy as np
import torch
import torch.distributed as dist


class LatLon(NamedTuple):
    lat: float
    lon: float

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


class TimeLatLon(NamedTuple):
    time: int
    lat: int
    lon: int

    def as_latlon(self):
        return LatLon(self.lat, self.lon)


class Coords:
    def __init__(self, start, bound, stepsize, extra=None):
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

    def get_all(self):
        return np.array([self.from_index(i) for i in range(self.n_tiles)])

    @property
    def has_extra(self):
        return not self._extra is None


class DistributedSamples:
    def __init__(
        self,
        token_sizes: TimeLatLon,  # field[i][4]
        num_tokens: TimeLatLon,  # field[i][3]
        res: float,
        token_overlap=LatLon(*[0, 0]),
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.res = res # 360.0 / file_shape.lon  # assumes same lat/lon res
        self.token_sizes = token_sizes
        self.num_tokens = num_tokens
        self.token_overlap = token_overlap

        self.lats, self.lons = self.get_coords()

        self.n_samples
        self.batchsize

    def get_coords(self):
        # assumed that sanity checking that field data is consistent has been done
        token_len = self.token_sizes.as_latlon() * self.res
        side_len = self.num_tokens.as_latlon() * token_len
        overlap = self.token_overlap * token_len
        half_side_len = side_len / 2.0

        assert (
            overlap <= half_side_len
        ), "token_overlap too large for #tokens, reduce if possible"

        start = LatLon(half_side_len.lat, half_side_len.lon - overlap.lon / 2.0)
        bound = LatLon(180 - half_side_len.lat, 360 + half_side_len.lon)
        step_size = side_len - overlap
        
        # add one additional row if no perfect tiling (sphere is toric in longitude so no special
        # handling necessary but not in latitude)
        # the added row is such that it goes exaclty down to the South pole and the offset North-wards
        # is computed based on this
        needs_extra_row = (lats.n_tiles - 1) * (
            lats.stepsize
        ) - half_side_len.lat < 180.0
        extra_row_lat = 180.0 - half_side_len.lat + self.res
        
        extra = extra_row_lat if needs_extra_row else None
        
        lats = Coords.from_range(start.lat, bound.lat, step_size.lat, extra=extra)
        lons = Coords.from_range(start.lon, bound.lon, step_size.lon)

        return lats, lons
    

    def global_times_pos(self, times):
        """generate patch/token positions for global grid."""

        # generate tiles
        times_pos = []
        for time in times:
            for lat, lon in it.product(self.lats.get_all(), self.lons.get_all()):
                times_pos.append([*time, -lat + 90.0, np.mod(lon, 360.0)])

            print(f"ctime: {time}, n_times_pos: {len(times_pos)}")


def as_global_batch_idx(task_batch_idx: int) -> int:
    pass


def as_task_batch_idx(global_batch_idx) -> int:
    pass


def daterange() -> tuple[datetime, datetime]:
    pass
