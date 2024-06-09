from abc import ABC, abstractmethod
import datetime as dt
from itertools import product, chain
from collections.abc import Iterable

from atmorep.datasets.parallel_data_writer import XarrayWriter
from atmorep.datasets.sampling import GlobalSamples, LocationSamples
from atmorep.utils.dataclasses import OutputType
import numpy as np
import pandas as pd

from atmorep.core.evaluator import Evaluator
from atmorep.utils.utils import NetMode
from atmorep.core.forecasting import GlobalForecastRunner
from atmorep.datasets.output import InferenceDataAdapter, LogGlobalForecast, Output, OutputConfiguration, WriterConfiguration

class Global(Evaluator, ABC):
    def prepare_model(self):
        dates = self.get_dates()
        self.evaluator.model.set_sampling(
            NetMode.test,
            GlobalSamples.from_config(self.config, np.array(dates))
        )
        
        output_config = OutputConfiguration(
            out_dataset,
            {OutputType.pred},
            XarrayWriter
        )
        
        
        self.add_output(output_config.construct)
        
  
    @abstractmethod
    def get_dates(self, args):
        pass


class Validation(Evaluator, ABC):
    def run_model(self):
        self.evaluator.validate(0, self.config.BERT_strategy)


class Evaluation(Evaluator, ABC):
    def run_model(self):
        self.evaluator.evaluate( 0, self.config.BERT_strategy)

class Inference(Evaluator, ABC):
    def run_model(self):
        self.evaluator.predict(self.config.BERT_strategy)

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
    
class MyGlobalForecastNetCDF(Inference, Global, mode="global_forecast"):
    @classmethod
    def get_evaluator(cls, config, model_id, model_epoch, devices):
        
        runner = GlobalForecastRunner.load(config, model_id, model_epoch, devices)
        return runner

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
    
class GlobalForecastMonths(MyGlobalForecastNetCDF, mode="global_forecast_months"):
    def get_dates(self):
        datetimes = []
        for year, month in self.config.months_inference:
            datetimes.extend(
                self.get_dates_in_month(year, month, self.config.lead_time)
            )

        return datetimes

    def get_dates_in_month(
        self, year: int, month: int, lead_time: int
    ) -> Iterable[dt.datetime]:
        start = dt.datetime(year, month, 1)
        return self.get_dates_from_start_freq(start, "M", lead_time)

    @staticmethod
    def get_dates_from_start_freq(
        start, freq, lead_time=3
    ):  # freq is "M" for month "Y" for years
        start, end = pd.period_range(start, periods=2, freq=freq)
        index = pd.period_range(start.to_timestamp(), end.to_timestamp(), freq="6h")[
            :-1
        ] - dt.timedelta(hours=lead_time)

        return [[stamp.year, stamp.month, stamp.day, stamp.hour] for stamp in index]

class TemporalInterpolation(Validation, mode = "temporal_interpolation"):
    @classmethod
    def get_config_options(cls):
        return dict(
            BERT_strategy = 'temporal_interpolation',
            log_test_num_ranks = 4
        )


class GlobalForecast(Validation, Global, mode="global_forecast_original"):
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

class GlobalForecastYears(MyGlobalForecastNetCDF, mode="global_forecast_years"):
    def get_dates(self):
        datetimes = []
        for year in self.config.years_inference:
            datetimes.extend(self.get_dates_in_year(year, self.config.lead_time))

        return datetimes

    def get_dates_in_year(self, year: int, lead_time: int) -> Iterable[dt.datetime]:
        start = dt.datetime(year, 1, 1)
        return self.get_dates_from_start_freq(start, "Y", lead_time)



class IconCmip(Evaluator, ABC):
    @classmethod
    def get_config_options(cls):
        return dict( # overwrite default data_dir path => load different dataset
            data_dir = "./data/icon_cmip"
        )


class CmipGlobalForecast(MyGlobalForecastNetCDF, IconCmip, mode="cmip_global_forecast"):
    pass


class CmipGlobalForecastMonths(
    GlobalForecastMonths, IconCmip, mode="cmip_global_forecast_months"
):
    pass


class CmipGlobalForecastYears(
    GlobalForecastYears, IconCmip, mode="cmip_global_forecast_years"
):
    pass
    
    

class GlobalForecastRange(Evaluation, Global, mode = "global_forecast_range"):
    def get_dates(self):
        dates = [ ]
        num_steps = 31*2 
        cur_date = [2018, 1, 1, 0+6] #6h models
        for _ in range(num_steps) :
            tdate = dt.datetime(
                cur_date[0], cur_date[1], cur_date[2], cur_date[3]
            )
            tdate += dt.timedelta( hours = 12 )
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
    
        sampling = LocationSamples(pos, years, months, num_t_samples_per_month)
        self.evaluator.model.set_sampling(NetMode.test, sampling)
    
    def run_model(self):
        self.evaluator.evaluate( 0)