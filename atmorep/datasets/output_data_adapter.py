from typing import Any, Type

from atmorep.datasets import output
import numpy as np

import atmorep.utils as utils
from atmorep.utils.dataclasses import Dimensions, TimeLatLon, OutputType
from atmorep.datasets.normalizer_local import NormalizerLocal
from atmorep.datasets.output import Field

class InferenceDataAdapter:
    _adapters: dict[OutputType, Type["InferenceDataAdapter"]] = dict()
    
    def __init__(
        self,
        tail_num_nets: int,
        n_levels: int,
        forecast_n_tokens: int,
        n_tokens: TimeLatLon,
        token_size: TimeLatLon,
        normalizers: dict[tuple[str, int], NormalizerLocal]
    ):
        self.tail_num_nets = tail_num_nets
        
        self.n_levels = n_levels
        self.forecast_n_tokens = forecast_n_tokens
        self.n_tokens = n_tokens
        self.token_sizes = token_size
        
        self.normalizers = normalizers
    
    def __init_subclass__(cls, output_type, **kwargs) -> None:
        super().__init__subclass__(**kwargs)
        InferenceDataAdapter._adapters[output_type] = cls
    
    @classmethod
    def from_config(
        cls,
        config: Any,
        output_type: OutputType,
        normalizers: dict[tuple[str, int], NormalizerLocal]
    ):
        # TODO: maybe field dependent ???
        field_idx = 0 # default first field
        adapter_class = cls._adapters[output_type]
        return adapter_class(
            config.net_tail_num_nets,
            len(config.fields[field_idx][2]),
            config.forecast_num_tokens,
            TimeLatLon(*config.fields[field_idx][4]),
            TimeLatLon(*config.fields[field_idx][3]),
            normalizers
        )
    
    def reshape(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(
            self.n_levels,
            -1,
            self.forecast_num_tokens,
            self.n_tokens.lat,
            self.n_tokens.lon,
            self.token_sizes.time,
            self.token_sizes.lat,
            self.token_sizes.lon
        )
    
    def detokenize_field(self, data: np.ndarray) -> np.ndarray:
        """get datacube from flat vector.
        
        Args:
            data: Flat data output from atmorep.
        
        Returns:
            Datacube with shape:
            (
                (net_tail_num_nets,) n_samples,
                sample_size_vertical,
                sample_size_time,
                sample_size_lat, 
                sample_size_lon
            )
        """
        return utils.detokenize(
            self.reshape(data)
        ).swapaxes(0, 1)
    
    def denormalize_sample(
        self,
        field_sample: Field,
        sample_coords: dict[Dimensions, np.ndarray],
        lvl: int,
    ) -> Field:
        """
        Process atmorep output into datacube.
        
        Returns:
            Denormalized data cube.
        """
        normalizer = self.normalizers[field_sample.name, lvl]
        # should be only 1 value
        date = sample_coords[Dimensions.datetime][0]
        lat_lon = (
            sample_coords[Dimensions.lat],
            sample_coords[Dimensions.lon]
        )
        normalized_data = normalizer.denormalize(
            date.year, date.month, field_sample.data, lat_lon
        )
        
        return Field(field_sample.name, normalized_data)

class Source(InferenceDataAdapter, output_type=OutputType.source):
    def reshape(self, data):
        pass
        
class Target(InferenceDataAdapter, output_type=OutputType.target):
    pass

class Prediction(InferenceDataAdapter, output_type=OutputType.pred):
    pass

class Ensemble(InferenceDataAdapter, output_type=OutputType.ens):
    def reshape(self, data):
        return data.reshape(
            self.net_tail_num_nets,
            self.n_levels,
            -1,
            self.forecast_num_tokens,
            self.n_tokens.lat,
            self.n_tokens.lon,
            self.token_sizes.time,
            self.token_sizes.lat,
            self.token_sizes.lon
        ).swapaxes(1, 2)
