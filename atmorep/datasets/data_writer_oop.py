from abc import ABC, abstractmethod
from enum import StrEnum

import zarr


class OutputType(StrEnum):  # needs python 3.11
    source = "source"
    target = "target"
    pred = "pred"
    ens = "ens"


class InferenceDataAdapter(ABC):
    def __init__(self, output_type: OutputType, store, subset_start_batch_idx=0):
        self.output_type = output_type
        self.store = store
        self.subset_start_batch_idx = subset_start_batch_idx

    def write(
        self,
        fields,
        batch_idx,
        levels,
        coords,
    ):
        root_group = zarr.group(store=self.store.store)
        for fidx, field in enumerate(fields):
            field_name, data = field

            # for BERT: skip fields that were not predicted
            if not self.skip_data(data):
                var_subgroup = root_group.require_group(f"{field_name}")
                batch_size = data.shape[0]
                for sample_idx in range(batch_size):
                    global_sample_idx = self.as_global_idx(
                        batch_idx, batch_size, sample_idx
                    )
                    sample_subgroup = var_subgroup.create_group(
                        f"sample={global_sample_idx:05d}"
                    )
                    self.write_sample(
                        sample_subgroup, fidx, sample_idx, data, levels, coords
                    )

    def as_global_idx(self, batch_idx, batch_size, sample_idx: int):
        return (self.subset_start_batch_idx + batch_idx) * batch_size + sample_idx

    @abstractmethod
    def write_sample(sample_group, fidx, bidx, data, levels, coords):
        pass

    def skip_data(self, data):
        return False


class WriterForecast(InferenceDataAdapter):
    def write_sample(self, sample_group, fidx, bidx, data, levels, coords):
        time, lat, lon = coords
        sample_group.create_dataset("data", data=data[bidx])
        sample_group.create_dataset("ml", data=levels)
        sample_group.create_dataset("datetime", data=time[bidx])
        sample_group.create_dataset("lat", data=lat[bidx])
        sample_group.create_dataset("lon", data=lon[bidx])


class WriterForecastBERT(InferenceDataAdapter):
    def write_sample(self, sample_group, fidx, bidx, data, levels, coords):
        time, lat, lon = coords[:3]
        sample_group.create_dataset("data", data=data[bidx])
        sample_group.create_dataset("ml", data=levels[fidx])
        sample_group.create_dataset("datetime", data=time[0][bidx])
        sample_group.create_dataset("lat", data=lat[0][bidx])
        sample_group.create_dataset("lon", data=lon[0][bidx])


class WriterBERTVertical(InferenceDataAdapter):
    def write_sample(self, sample_group, fidx, bidx, data, levels, coords):
        time, lat, lon = coords[:3]
        for vidx in range(len(levels[fidx])):
            m_lvl = sample_group.require_group(f"ml={levels[fidx][vidx]}")
            m_lvl.create_dataset("data", data=data[vidx][bidx])
            m_lvl.create_dataset("ml", data=levels[fidx][vidx])
            m_lvl.create_dataset("datetime", data=time[fidx][bidx][vidx])
            m_lvl.create_dataset("lat", data=lat[fidx][bidx][vidx])
            m_lvl.create_dataset("lon", data=lon[fidx][bidx][vidx])

    def skip_data(self, data):
        return (not self.output_type == OutputType.source) and (len(data[0]) == 0)
