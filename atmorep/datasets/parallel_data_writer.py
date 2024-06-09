from datetime import datetime
from copy import deepcopy
from pathlib import Path


from atmorep.datasets.output_data_adapter import InferenceDataAdapter
import numpy as np
import zarr
import xarray as xr

import torch.distributed as dist

import atmorep.config.config as config
from atmorep.utils.dataclasses import Dimensions, SpatialDimensions
from atmorep.datasets.sampling import TimeChunkedDataset

DEFAULT_DATA_AXES = tuple(*SpatialDimensions)

class GlobalFieldsBuffer:
    def __init__(
        self,
        fields: list[str],
        spatial_coords: dict[Dimensions, np.ndarray],
        tile_axes_size: dict[Dimensions, int],
        data_axes: tuple[SpatialDimensions] = DEFAULT_DATA_AXES
    ):
        self.fields = fields
        self.spatial_coords = spatial_coords

        self.data_axes = data_axes
        self.tile_axes_size = tile_axes_size
        self.shape = tuple(
            1, *[self.spatial_coords[dim].size for dim in self.data_axes]
        )
        self.tile_shape = tuple(
            1, *[self.tile_axes_size[dim] for dim in self.data_axes]
        )
        
        self.check_compatibility(self.shape, self.tile_shape)
        
        self.spatial_coords = spatial_coords
        self.buffer = {field: np.empty(self.shape, dtype=float) for field in fields}
        
        # mask already used coords
        self.spatial_coords_mask = self.generate_coords_mask()
        self.current_time = None
    
    @staticmethod
    def check_compatibility(global_shape: tuple[int], tile_shape: tuple[int]):
        """Assure that the global domain gets tiled perfectly."""
        if not len(global_shape) == len(tile_shape):
            raise ValueError(f"incompatible shape {tile_shape} to tile global domain with shape {global_shape} perfectly.")
        
        for global_dim_size, tile_dim_size in zip(global_shape, tile_shape):
            if not global_dim_size % tile_dim_size == 0:
                raise ValueError(f"incompatible shape {tile_shape} to tile global domain with shape {global_shape} perfectly.")

    def generate_coords_mask(self):
        return {
            dim: np.zeros_like(coord, dtype=bool)
            for dim, coord in self.spatial_coords.items()
        }

    def write_sample(self, fields, coords: dict[Dimensions, np.ndarray]):
        hypercube = self.get_slices(coords)
        for field_name, field_data in fields.items():
            self.buffer[field_name][hypercube] = field_data
    
    def get_slices(self, coords: dict[Dimensions, np.ndarray]) -> tuple[slice]:
        slices = []
        for dimension in self.data_axes:
            dim_slice = self.get_slice(coords[dimension], dimension)
            if not any(self.spatial_coords_mask[dimension][dim_slice]):
                slices.append(dim_slice)
                self.spatial_coords_mask[dimension][dim_slice] = 1
            else:
                raise ValueError(f"attempt to overwrite existing chunk at dim {dimension}:{dim_slice}")
        
        return tuple(slices)
    
    def get_slice(self, coords: np.ndarray, dimension: SpatialDimensions) -> slice:
        expected_size = self.tile_axes_size[dimension]
        if coords.size == expected_size:
            start_index = self.as_index(coords[0], dimension)
            end_index = self.as_index(coords[-1], dimension)
    
            if not end_index - start_index == expected_size:
                raise ValueError(f"tile dimension coords {coords} dont align with global grid")
            
            return slice(start_index, end_index)        
        else:
            raise ValueError(f"tile dimension {dimension} size {coords.size} doesnt match expected size {expected_size}")
    
    def as_index(self, coordinate: float, dimension: SpatialDimensions) -> int:
        start = self.spatial_coords[dimension][0]
        # expect to be difference to be the same everywhere
        step = np.diff(self.spatial_coordss[dimension][:1])[0]
        index = int((coordinate - start) // step)
        if not int((coordinate - start) % step) == 0:
            raise ValueError(f"coordinate value {coordinate} is not aligned with global grid.")
        
        return index
    
    def as_dataset(self):
        coords = self.spatial_coords | {Dimensions.datetime: np.array(np.datetime64(self.current_time))}
        fields = {
                field_name: (
                    self.dims,
                    self.buffer[field_name]
                )
                for field_name in self.fields
            }
        return xr.Dataset(coords, data_vars = fields)
    
    @property
    def is_full(self):
        return all(all(coords) for coords in self.spatial_coords_mask.values())
    
    def setup_next_time_chunk(self, time: datetime):
        """reset buffer and update coords."""
        self.current_time = time
        self.spatial_coords_mask = deepcopy(self.spatial_coords)


class XarrayWriter:
    def __init__(
        self,
        data_adapter: InferenceDataAdapter,
        dataset: TimeChunkedDataset,
        store_path: Path
    ):
        super().__init__(self, data_adapter, dataset)
        self.dataset = dataset
        self.store_path = store_path
        
        # times unknown during initialization => intatiate on first call
        self._store = None
        self._fields_buffer = None

    @classmethod
    def from_config(cls, config):
        dataset = TimeChunkedDataset.from_config(config)
        
    @staticmethod
    def get_store(store_path):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        substore_pathes = XarrayWriter.get_store_pathes(
            store_path, world_size
        )
        return zarr.ZipStore(substore_pathes[rank], mode="w")

    @staticmethod
    def get_store_pathes(stores_path, n_tasks):
        return [stores_path / f"rank_{n}.zip" for n in range(n_tasks)]

    @property
    def store(self):
        # dataset time dimenion is not available at instantiation time
        if self._store is None:
            self._store = self.get_store(self.store_path)
            # imitade actual xarray dataset
            self.dataset.to_zarr(self.store, compute=False)

        return self.store

    @property
    def fields_buffer(self):
        if self._fields_buffer is None:
            self._fields_buffer = GlobalFieldsBuffer(self.dataset.spatial_coords)
            self.reset_buffer() # get starting time
        
        return self._fields_buffer

    def write_sample(
        self,
        batch_idx,
        fields: dict[str, np.ndarray],
        coords: dict[Dimensions, np.ndarray]
    ):
        """
        write results for one batch of sample.
        
        fields: contains data variables var_name: var_data
                with shape (sample_idx, lvl, time, lat, lon)
        coords: coordinate variables for var_data, should match in shape
        """
        self.fields_buffer.write_sample(fields, coords)
        
        if self.fields_buffer.is_filled:
            self.write_buffer()
            
    
    def write_buffer(self):
        ds = self.fields_buffer.as_dataset()
        ds.to_zarr(self.store, region=ds.coords) # be carefull with region arg
        
        self.store.flush()    
    
    def reset_buffer(self):
        next_time = self.dataset.get_next_chunk_time(self.fields_buffer.current_time)
        self.fields_buffer.setup_next_time_chunk(next_time)
        

def get_store_file_path(model_id, epoch, output_type):
    store_dir = get_store_dir(model_id)
    return store_dir / f"results_id{model_id}_epoch{epoch:05d}_{output_type}.zarr"


def get_store_dir(model_id):
    return config.PATHES.results / f"id{model_id}"
