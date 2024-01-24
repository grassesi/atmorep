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

from abc import ABC, abstractmethod, abstractclassmethod

import wandb

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config, setup_wandb, setup_hpc, NetMode

    
class Evaluator(ABC):
  mode = None
  _modes = dict()
  
  # registers every subclass that properly implements a mode
  def __init_subclass__(cls, mode=None, **kwargs) -> None:
     super().__init_subclass__(**kwargs)
     if mode is not None:
      Evaluator._modes[mode] = cls
  
  @staticmethod
  def evaluate(mode, model_id, args={}, model_epoch=-2):
    try:
      evaluation_type = Evaluator._modes[mode]
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
  def get_config_options(cls):
    pass
  
  def run(self):
    self.prepare_model()
    self.wandb_output()
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