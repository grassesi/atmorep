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

import argparse
import json

from atmorep.core.evaluator import Evaluator

if __name__ == '__main__':

  # models for individual fields
  # model_id = '4nvwbetz'     # vorticity
  # model_id = 'oxpycr7w'     # divergence
  # model_id = '1565pb1f'     # specific_humidity
  # model_id = '3kdutwqb'     # total precip
  # model_id = 'dys79lgw'     # velocity_u
  # model_id = '22j6gysw'     # velocity_v
  # model_id = '15oisw8d'     # velocity_z
  # model_id = '3qou60es'     # temperature (also 2147fkco)
  # model_id = '2147fkco'     # temperature (also 2147fkco)

  # multi-field configurations with either velocity or voritcity+divergence
  # model_id = '1jh2qvrx'     # multiformer, velocity
  # model_id = 'wqqy94oa'     # multiformer, vorticity
  # model_id = '3cizyl1q'     # 3 field config: u,v,T
  # model_id = '1v4qk0qx'     # pre-trained, 3h forecasting
  # model_id = '1m79039j'     # pre-trained, 6h forecasting
  
  # supported modes: test, forecast, fixed_location, temporal_interpolation, global_forecast,
  #                  global_forecast_range
  # options can be used to over-write parameters in config; some modes also have specific options, 
  # e.g. global_forecast where a start date can be specified
  
  # BERT masked token model
  # mode, options = 'BERT', {'years_test' : [2021], 'fields[0][2]' : [123, 137], 'attention' : False}
  
  # BERT forecast mode
  # mode, options = 'forecast', {'forecast_num_tokens' : 1} #, 'fields[0][2]' : [123, 137], 'attention' : False }
  
  # BERT forecast with patching to obtain global forecast
  
  parser = argparse.ArgumentParser(
    prog="Atmorep-evaluate",
    description="run atmorep for inference/evaluation"
  )
  parser.add_argument("mode", help="evaluation mode")
  parser.add_argument("model_id", help="trained model to be used.")
  parser.add_argument("options_file", help="json file for additional configuration")
  
  args = parser.parse_args()
  
  with open(args.options_file, "r") as f:
    options = json.load(f)

  print(f"evaluating with {args.mode}, {args.model_id}, {options}")
  Evaluator.evaluate(args.mode, args.model_id, args=options)
