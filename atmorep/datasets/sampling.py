from dataclasses import dataclass
from itertools import product
from abc import ABC, abstractmethod
from datetime import datetime


from atmorep.datasets.dataset import Dataset
import numpy as np
import torch.distributed as dist

from atmorep.utils.utils import days_in_month
from atmorep.utils.dataclasses import Coords, FileParams, LatLon, TokenParams


    
class DistributionParams:
    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        batches_per_date: int,
    ):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.batches_per_date = batches_per_date

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # should always divide evenly
        self.n_batches = self.n_samples // self.batch_size

        self.n_dates = self.n_batches // self.batches_per_date
        self.dates_per_task = self.n_dates // self.world_size

        self._batch_range = None
        self._sample_range = None

    @property
    def batch_range(self):
        if self._batch_range is None:
            self._batch_range = self.get_batch_range()
        return self.get_batch_range

    def get_batch_range(self):
        # should divide without remainder
        batches_per_task = self.dates_per_task * self.batches_per_date

        batches_per_task = self.n_batches // self.world_size
        n_rest_tasks = self.n_batches % self.world_size

        start = self.get_batch_start(self.rank, batches_per_task, n_rest_tasks)
        stop = self.get_batch_start(self.rank + 1, batches_per_task, n_rest_tasks)

        return np.arange(start, stop)

    @property
    def sample_range(self):
        if self._sample_range is None:
            samples_start = self.batch_range * self.batch_size
            self._sample_range = np.arange(samples_start[0], samples_start[-1])
        return self._sample_range

    @property
    def batch_start(self):
        return self.batch_range[0]

    @property
    def batch_end(self):
        return self.batch_end[-1]

    @property
    def sample_start(self):
        return self.sample_range[0]

    @property
    def sample_end(self):
        return self.sample_range[-1]

    @staticmethod
    def get_batch_start(rank: int, batches_per_task: int, n_rest_tasks: int):
        gets_rest_task = rank < n_rest_tasks

        if gets_rest_task:
            return rank * batches_per_task + rank
        else:
            return rank * batches_per_task + n_rest_tasks

    def __str__(self):
        f"assignment rank {self.rank}: batch {self.batch_start}-{self.batch_end}/{self.n_batches} (sample {self.sample_start}-{self.sample_end})"


class DistributedSamples(ABC):
    def __init__(self, batch_size):
        self._time_pos = None
        self.batch_size = batch_size
        self.n_samples = len(self)

    @property
    def times_pos(self):
        if self._time_pos is None:
            self._time_pos = self.get_time_pos()

        return self._time_pos

    @abstractmethod
    def get_time_pos(self):
        return

    @abstractmethod
    def __len__(self):
        pass


# self.total_n_batches = len(times_pos) // batch_size
# self.subset_start_batch_idx = batch_start
# assigned_samples = times_pos[sample_start:sample_end]

class RandomSamples(DistributedSamples):
    def __init__(self, dataset):
        pass
    
    def get_time_pos(self):
        pass
    
    def __len__(self):
        pass

class GlobalSamples(DistributedSamples):
    def __init__(
        self,
        tokens: TokenParams,
        file_params: FileParams,
        dataset: Dataset
    ):
        self.tokens = tokens
        self.file_params: file_params

        self.lats, self.lons = self.get_coords()

        # n_samples = self.lats.n_tiles * self.lons.n_tiles
        super().__init__(self.lons.n_tiles)
    
    @classmethod    
    def from_config(cls, config):
        tokens = TokenParams.from_field(0, config)
        file_params = FileParams.from_config(config)
        
        return cls(tokens, file_params)

    def get_coords(self) -> tuple[Coords]:
        # assumed that sanity checking that field data is consistent has been done
        side_len, overlap = self.tokens.get_sample_len_overlap(self.file_params)
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
        extra_row_lat = 180.0 - half_side_len.lat + self.file_params.resolution

        extra = extra_row_lat if needs_extra_row else None

        lats = Coords.from_range(start.lat, bound.lat, step_size.lat, extra=extra)
        lons = Coords.from_range(start.lon, bound.lon, step_size.lon)

        return lats, lons

    def get_times_pos(self):
        """generate patch/token positions for global grid."""

        # generate tiles
        times_pos = []
        for time in self.times:
            for lat, lon in product(self.lats.get_all(), self.lons.get_all()):
                times_pos.append([*time, -lat + 90.0, np.mod(lon, 360.0)])

            print(f"ctime: {time}, n_times_pos: {len(times_pos)}")

        return times_pos
    
    def get_batch_start(batch_idx: int) -> int: # TODO
        """calculates global index of first sample in batch."""
        pass

    def __len__(self):
        return self.lats.n_tiles * self.lons.n_tiles * len(self.times)
    


class LocationSamples(DistributedSamples):
    def __init__(
        self, pos, years, months, num_t_samples_per_month, batch_size, rng_seed=None
    ):
        self.pos = pos
        self.years = years
        self.months = months

        self.num_t_samples_per_month = num_t_samples_per_month
        self.batch_size = batch_size

        if rng_seed is None:
            rng_seed = np.random.randint(0, 100000, 1)[0]
        self.rng = np.random.default_rng(rng_seed)

    def get_times_pos(self):
        """random time sampling for fixed location"""

        times_pos = []
        for i_ym, ym in enumerate(product(self.years, self.months)):

            # ensure a constant size of work load of data loader independent of the month length
            # factor of 128 is a fudge parameter to ensure that mod-ing leads to sufficiently
            # random wrap-around (with 1 instead of 128 there is clustering on the first days)
            hours_in_day = int(24 / self.time_sampling)
            d_i_m = days_in_month(ym[0], ym[1])
            perms = self.rng.permutation(self.num_t_samples_per_month * d_i_m)
            # ensure that days start at 1
            perms = np.mod(perms[: self.num_t_samples_per_month], (d_i_m - 1)) + 1
            rhs = self.rng.integers(
                low=0, high=hours_in_day, size=self.num_t_samples_per_month
            )

            for rh, perm in zip(rhs, perms):
                times_pos += [[ym[0], ym[1], perm, rh, self.pos[0], self.pos[1]]]

        # adjust batch size if necessary so that the evaluations split up across batches of equal size
        while 0 != (len(times_pos) % batch_size):
            batch_size -= 1
        assert batch_size >= 1

        return times_pos

    def __len__(self):
        len(list(product(self.years, self.months))) * self.num_t_samples_per_month
