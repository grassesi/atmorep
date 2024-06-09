from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from atmorep.core.atmorep_model import AtmoRepData
from atmorep.datasets.dataset import Dataset
from atmorep.datasets.output_data_adapter import InferenceDataAdapter
from atmorep.datasets.parallel_data_writer import XarrayWriter
import numpy as np


from atmorep.utils.dataclasses import Dimensions, Field, OutputData, OutputType
from atmorep.datasets.sampling import DistributedSamples

class Writer(ABC):
    def __init__(self, data_adapter: InferenceDataAdapter, dataset: Dataset):
        self.data_adapter = data_adapter
        self.dataset = dataset
    
    def write(
        self,
        batch_start_idx: int,
        output_data: OutputData
    ):
        for field in output_data.fields:
            data = self.data_adapter.detokenize_field(field.data)
            
            for sample_idx, (sample, sample_coords) in enumerate(
                zip(data, output_data.coords)
            ):
                for lvl in sample_coords[Dimensions.lev]:
                    global_sample_idx = (
                        batch_start_idx + sample_idx
                    )
                    sample_data = self.data_adapter.denormalize(
                        Field(field.name, sample), sample_coords, lvl
                    )
                    
                    self.write_sample(
                        global_sample_idx, sample_data, sample_coords
                    )
    
    
    @abstractmethod
    def write_sample(
        self,
        global_sample_idx: int,
        field: Field,
        coords: dict[Dimensions, np.ndarray]
    ):
        pass

class Output():
    def __init__(
        self, field_names: set[str], writers: dict[OutputType, Writer]
    ):
        self.field_names = field_names
        self.writers = writers
        
        self.dataset = self.writers[OutputType.target].dataset # FIXME
    
    @classmethod
    def from_config(cls, config, writers: dict[OutputType, Writer]):
        field_names = {field_info[0] for field_info in config.fields}
        
        return cls(field_names, writers)


    def log(self, batch_start_idx: int, log_sources, log_preds):
        # save source: remains identical so just save ones
        
        input_data = {
            OutputType.source: log_sources[0],
            OutputType.target: log_sources[2],
            OutputType.pred: log_preds[0],
            OutputType.ens: log_preds[2],
        }
        
        output_data = {
            OutputType.source: OutputData([], coords_source),
            OutputType.target: OutputData([], coords_targets),
            OutputType.pred: OutputData([], coords_targets),
            OutputType.ens: OutputData([], coords_targets)
        }

        token_infos = log_sources[1][0] # use token infos from first field
        coords_source = self.dataset.get_source_coords(token_infos)
        coords_targets = self.dataset.get_target_coords(token_infos)
        for output_type in OutputType:
            for field, name in zip(input_data[output_type], self.field_names):
                output_data[output_type].fields.append(
                    Field(name, field.cpu().detach().numpy())
                ) 
        
        self.write_files(batch_start_idx, output_data)

    def set_sampling(samples: DistributedSamples):
        pass # TODO
    
    def write_files(
        self,
        batch_start_idx,
        output_data: dict[OutputType, OutputData],
    ):
        for output_type, writer in self.writers.items():
            writer.write(batch_start_idx, output_data[output_type])


@dataclass
class OutputConfiguration:
    output_dataset: Dataset
    output_types: set[OutputType]
    writer_type: Type[Writer]
    
    def construct(self, atmorep_model: AtmoRepData) -> Output:
        normalizers = atmorep_model.get_all_normalizers(field_names, vlvls)
        data_adapter = InferenceDataAdapter.from_config(
            self.config, OutputType.pred, normalizers
        )
        writers = {OutputType.pred: XarrayWriter(data_adapter, out_dataset, store_path)}
        output = Output.from_config(self.config, writers)