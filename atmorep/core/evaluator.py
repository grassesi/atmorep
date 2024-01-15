####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import datetime
from abc import ABC, abstractmethod, abstractclassmethod

import numpy as np
import wandb

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config, setup_wandb, setup_hpc, NetMode

    
class Evaluator(ABC):
  mode = None
  _modes = dict()
  
  # registers every subclass that properly implements a mode
  def __new__(cls, name, bases, attrs):
    new_cls = super().__new__(cls, name, bases, attrs)
    if new_cls.mode is not None: # new_cls implements a mode
      cls._modes[new_cls.mode] = new_cls
    
    return new_cls
  
  @classmethod
  def evaluate(cls, mode, model_id, args={}, model_epoch=-2):
    try:
      evaluation_type = cls._modes[mode]
      evaluation_mode = evaluation_type(model_id, model_epoch, args)
      evaluation_mode.run()
    except KeyError as e:
      print(f"no such evaluation mode: {mode}")
    
  
  def __init__(self, model_id, model_epoch, args):
    config, devices = Evaluator.setup(model_id)
    
    config.add_args(self.get_config_options())
    config.add_args(args)
    self.config = config
    self.evaluator = Trainer_BERT.load(
      self.config, model_id, model_epoch, devices
    )
  
  @classmethod
  def setup(cls, model_id):
    config = Config(wandb_id=model_id)
    
    with_ddp, par_rank, par_size, devices = setup_hpc()
    # overwrite old config
    config.add_args(
      cls.get_hpc_options(
        with_ddp, par_rank, par_size, config.loader_num_workers
        )
    )
    
    setup_wandb(config.with_wandb, config, par_rank, '', mode='offline')
    config.attention = False
    
    if config.par_rank == 0:
      print( 'Running Evaluate.evaluate with mode =', cls.mode)
    
    return config, devices
  
  @staticmethod
  def get_hpc_options(with_ddp, par_rank, par_size, loader_num_workers):
    return dict(
      with_wandb = True,
      with_ddp = with_ddp,
      par_rank = par_rank,
      par_size = par_size,
      num_loader_workers = loader_num_workers,
      data_dir = './data/',
    )
  
  @abstractclassmethod
  def get_mode(cls):
    pass
    
  @abstractclassmethod
  def get_config_options(cls):
    pass
  
  def run(self):
    self.prepare_model()
    self.wandb_output(self.config)
    self.run_model()

  def prepare_model(self):
    self.evaluator.model.load_data(NetMode.test)  
  
  def wandb_output(self):
    if self.config.par_rank == 0:
      self.config.print()
      self.config.write_json(wandb)
  
  @abstractmethod
  def run_model(self):
    pass


class Global(Evaluator, ABC):
  def prepare_model(self):
    dates = self.get_dates()
    self.evaluator.model.set_global(NetMode.test, np.array(dates))
  
  @abstractmethod
  def get_dates(self, args):
    pass


class Validation(Evaluator):
  def run_model(self):
    self.evaluator.validate(0, self.config.BERT_strategy)


class Evaluation(Evaluator):
  def run_model(self):
    self.evaluator.evaluate( 0, self.config.BERT_strategy)

    
class BERT(Validation):
  @classmethod
  def get_config_options(cls):
    return dict(
      lat_sampling_weighted = False,
      BERT_strategy = 'BERT',
      log_test_num_ranks = 4
    )

  
class Forecast(Validation):
  mode = "forecast"
  
  @classmethod
  def get_config_options(cls):
    return dict(
      lat_sampling_weighted = False,
      BERT_strategy = 'forecast',
      log_test_num_ranks = 4,
      forecast_num_tokens = 1  # will be overwritten when user specified
    )
    
    
class TemporalInterpolation(Validation):
  mode = "temporal_interpolation"
  
  @classmethod
  def get_config_options(cls):
    return dict(
      BERT_strategy = 'temporal_interpolation',
      log_test_num_ranks = 4
    )


class GlobalForecast(Validation, Global):
  mode = "global_forecast"
  
  @classmethod
  def get_config_options(cls):
    return dict(
      BERT_strategy = 'forecast',
      batch_size_test = 24,
      num_loader_workers = 1,
      log_test_num_ranks = 1
    )
  
  def get_dates(self):
    return self.config["dates"]


class GlobalForecastRange(Evaluation, Global):
  mode = "global_forecast_range"
  
  def get_dates(self):
    dates = [ ]
    num_steps = 31*2 
    cur_date = [2018, 1, 1, 0+6] #6h models
    for _ in range(num_steps) :
      tdate = datetime.datetime( cur_date[0], cur_date[1], cur_date[2], cur_date[3])
      tdate += datetime.timedelta( hours = 12 )
      cur_date = [tdate.year, tdate.month, tdate.day, tdate.hour]
      dates += [cur_date]
    
    return dates


class FixedLocation(Evaluator):
  mode = "fixed_location"
  
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
    
    self.evaluator.model.set_location( NetMode.test, pos, years, months, num_t_samples_per_month)
    
  def run_model(self):
    self.evaluator.evaluate( 0)