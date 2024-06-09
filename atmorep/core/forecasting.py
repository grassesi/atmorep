from random import sample
import torch

from atmorep.core.trainer import Trainer_BERT
from atmorep.datasets.output import Output
from atmorep.datasets.sampling import DistributedSamples
from atmorep.utils.utils import NetMode


class GlobalForecastRunner(Trainer_BERT):
    def __init__(self, config, devices):
        super().__init__(config, devices)
        self.output_writers: list[Output] = []

    def predict(self):
        """ "doesnt log attention, no wandb functionallity"""
        # BERT_strategy_train = self.cf.BERT_strategy
        # self.cf.BERT_strategy = BERT_test_strategy

        # self.model.set_mode(NetMode.test)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.model):
                global_idx = self.model.subset_start_batch_idx + batch_idx
                log_sources = self.get_log_sources(batch_data)
                prepared_batch_data = self.prepare_batch(batch_data)

                preds, atts = self.model(prepared_batch_data)

                # write predictions for all MPI-Tasks
                log_preds = [[p.detach().clone().cpu() for p in pred] for pred in preds]

                self.write_output(batch_idx, log_sources, log_preds)

        batch_data = []
        torch.cuda.empty_cache()

        # self.cf.BERT_strategy = BERT_strategy_train
        # self.mode_test = False

    def add_output(self, writer: Output):
        self.output_writers.append(writer)

    def set_sampling(self, mode: NetMode, samples: DistributedSamples):
        self.model.set_sampling(mode, samples)
        for writer in self.output_writers:
            writer.set_sampling(samples)

    def get_log_sources(self, batch_data):
        sources, token_infos, targets, tmis, tmis_list = batch_data
        log_sources = (
            [source.detach().clone().cpu() for source in sources ],
            [ti.detach().clone().cpu() for ti in token_infos],
            [target.detach().clone().cpu() for target in targets ],
            tmis,
            tmis_list
        )
        return log_sources
        
    def write_output(self, batch_idx: int, log_sources, log_preds):
        batch_start_sample_idx = self.model.dataset.samples.get_batch_start(
            batch_idx
        )
        for writer in self.output_writers:
            writer.log(batch_start_sample_idx, log_sources, log_preds)
