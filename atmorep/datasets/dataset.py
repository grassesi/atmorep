import itertools as it
from datetime import datetime

from atmorep.datasets.sampling import CoordParams
import torch
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

import atmorep.utils as utils
from atmorep.utils.dataclasses import Dimensions, LatLon, TimeLevLatLon, TokenInfo


class Dataset:
    def __init__(self, spatial_coords, samples):
        pass
    
    @classmethod
    def from_config(cls, config, times):
        pass
    
    def get_target_coords( # TODO
        self, batch_field_token_infos
    ) -> list[dict[Dimensions, np.ndarray]]:
        pass
    
    def get_source_coords( # TODO
        self, batch_field_token_infos
    ) -> list[dict[Dimensions, np.ndarray]]:
        n_samples = len(batch_field_token_infos)
        n_tokens_per_sample = len(batch_field_token_infos[0])
        n_tokens = n_tokens_per_sample * n_samples
        
        # chain token infos together from all samples in batch.
        records = it.chain(*batch_field_token_infos)
        
        columns=["year", "day", "hour", "vlvl", "lat", "lon", "vlvl_", "res"]
        batch_field_token_infos = pd.DataFrame.from_records(
            records, columns=columns, nrows=n_tokens
        )
        
        def as_datetime(token_info):
            return (
                utils.token_info_to_time(token_info)
                - pd.Timedelta(hours=self.time_token_size - 1)
            )
        
        batch_field_token_infos["datetime"] = batch_field_token_infos.apply(
            utils.token_info_to_time, axis=1
        )
        batch_field_token_infos.drop(["year", "day", "hour"])
        
        return batch_field_token_infos

    def extract_dates(self, field_batch_token_infos):
        # extract dates for each sample, constant for each batch and field
        time_token_size = self.token_params.size.time
        dates_t = np.empty(len(field_batch_token_infos[0]), dtype="datetime64")
        
        for sample_idx, tinfos in enumerate(field_batch_token_infos[0]):
            dates_t[sample_idx] = (
                utils.token_info_to_time(tinfos[0])
                - pd.Timedelta(hours=time_token_size - 1)
            )

        return dates_t

    def extract_vlvls(self, field_batch_token_infos):
        samples_info = field_batch_token_infos[0]
        samples_per_batch = len(samples_info)
        tokens_per_sample = len(samples_info[0])
        
        vlvls = np.empty(samples_per_batch*tokens_per_sample, dtype=int)
        for sample_idx, token_idx in it.product(
            range(samples_per_batch), range(tokens_per_sample)
        ):
            vlvl = samples_info[sample_idx][token_idx]
            vlvls[tokens_per_sample*sample_idx+token_idx] = vlvl
            
        return np.unique(vlvls)
    
    def reconstruct_geo_coords(self, field_batch_token_infos):
        # TODO enforce only coords from global axis
        # only valid because each batch exactly tiles in longitudes
        sample_info_north_west = field_batch_token_infos[0][0]
        sample_info_south_east = field_batch_token_infos[0][-1]
        
        south_west_token = TokenInfo(*sample_info_north_west[0])
        north_east_token = TokenInfo(*sample_info_south_east[-1])

        d_h = (self.token_params.size / 2).as_int()
        coords_min = LatLon(south_west_token.lat, south_west_token.lon)        
        coords_max = LatLon(north_east_token.lat, north_east_token.lon)
        
        start = coords_min - (d_h * self.file_params.resolution) + 0.001
        end = coords_max + (d_h * self.file_params.resolution) + 0.001
        
        lats = torch.arange(start.lat, end.lat, self.file_params.resolution).numpy()
        
        wrap_around = 360.0 if (coords_max.lon < coords_min.lon) else 0.0
        lons = torch.arange(start.lon, end.lon + wrap_around).numpy()
        lons = lons % 360
        
        return lats, lons
    

class TimeChunkedDataset:
    def __init__(self, field_names, coords, samples):
        samples.needs_samples(self)
        shape = TimeLevLatLon(*[coords[dim].size for dim in Dimensions])

        self._coords = None
        self.dims = [dim for dim in Dimensions]
        self.spatial_coords = {
            dim: self.coords[dim] for dim in [
                Dimensions.lat, Dimensions.lon, Dimensions.lev
            ]
        }

        self._chunked_dataset = None
    
    @property
    def coords(self):
        if self._coords is None:
            times = self.get_times(self.samples.get_samples)
            self._coords = self.spatial_coords | {Dimensions.time: times}
            
        return self._coords
    
    @property
    def chunked_dataset(self):
        if self._chunked_dataset is None:
            fields = {
                field_name: (
                    self.dims,
                    da.empty(shape=shape, chunks=(1, -1, -1, -1), dtype=float),
                )
                for field_name in field_names
            }
            xr.Dataset(coords, data_vars=fields)

    @classmethod
    def from_config(cls, config):
        sample_params = CoordParams.from_config(config)

        field_names = [field[0] for field in config.fields]
        coords = {dim: sample_params.get_global_coord(dim) for dim in Dimensions} # !!!
        return cls(field_names, coords)
    
    def get_next_chunk_time(current_time: datetime):
        pass
    
    def to_zarr(self, store, compute):
        self.chunked_dataset.to_zarr(store, compute=compute)
