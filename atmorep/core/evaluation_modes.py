from abc import ABC, abstractmethod
import datetime

import numpy as np

from atmorep.core.evaluator import Evaluator
from atmorep.utils.utils import NetMode

class Global(Evaluator, ABC):
    def prepare_model(self):
        dates = self.get_dates()
        self.evaluator.model.set_global(NetMode.test, np.array(dates))
  
    @abstractmethod
    def get_dates(self, args):
        pass


class Validation(Evaluator, ABC):
    def run_model(self):
        self.evaluator.validate(0, self.config.BERT_strategy)


class Evaluation(Evaluator, ABC):
    def run_model(self):
        self.evaluator.evaluate( 0, self.config.BERT_strategy)

    
class BERT(Validation):
    mode = "BERT"
    @classmethod
    def get_config_options(cls):
        return dict(
            lat_sampling_weighted = False,
            BERT_strategy = 'BERT',
            log_test_num_ranks = 4
        )

  
class Forecast(Validation, mode="forecast"):
    @classmethod
    def get_config_options(cls):
        return dict(
            lat_sampling_weighted = False,
            BERT_strategy = 'forecast',
            log_test_num_ranks = 4,
            forecast_num_tokens = 1  # will be overwritten when user specified
        )
    
    
class TemporalInterpolation(Validation, mode = "temporal_interpolation"):
    @classmethod
    def get_config_options(cls):
        return dict(
            BERT_strategy = 'temporal_interpolation',
            log_test_num_ranks = 4
        )


class GlobalForecast(Validation, Global, mode = "global_forecast"):
    @classmethod
    def get_config_options(cls):
        return dict(
            BERT_strategy = 'forecast',
            batch_size_test = 24,
            num_loader_workers = 1,
            log_test_num_ranks = 1
        )
  
    def get_dates(self):
        return self.config.dates


class GlobalForecastRange(Evaluation, Global, mode = "global_forecast_range"):
    def get_dates(self):
        dates = [ ]
        num_steps = 31*2 
        cur_date = [2018, 1, 1, 0+6] #6h models
        for _ in range(num_steps) :
            tdate = datetime.datetime(
                cur_date[0], cur_date[1], cur_date[2], cur_date[3]
            )
            tdate += datetime.timedelta( hours = 12 )
            cur_date = [tdate.year, tdate.month, tdate.day, tdate.hour]
            dates += [cur_date]
    
        return dates


class FixedLocation(Evaluator, mode = "fixed_location"):  
    @classmethod
    def get_config_options(cls):
        return dict(
            BERT_strategy = "BERT",
            num_files_test = 2,
            num_patches_per_t_test = 2,
            log_test_num_ranks = 4
        )
    
    def prepare_model(self):
        pos = [ 33.55 , 18.25 ]
        years = [2018]
        months = list(range(1,12+1))
        num_t_samples_per_month = 2
    
        self.evaluator.model.set_location(
            NetMode.test, pos, years, months, num_t_samples_per_month
        )
    
    def run_model(self):
        self.evaluator.evaluate( 0)