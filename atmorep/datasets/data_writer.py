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

import zarr

import atmorep.config.config as config

def write_batch_item_forecast(
  batch_item, fidx, bidx, data, levels, coords
):
  time, lat, lon = coords
  batch_item.create_dataset( 'data', data=data[bidx])
  batch_item.create_dataset( 'ml', data=levels)
  batch_item.create_dataset( 'datetime', data=time[bidx])
  batch_item.create_dataset( 'lat', data=lat[bidx])
  batch_item.create_dataset( 'lon', data=lon[bidx])

def write_batch_item_BERT(
  batch_item, fidx, bidx, data, levels, coords
):
  time, lat, lon = coords[:3]
  batch_item.create_dataset( 'data', data=data[bidx])
  batch_item.create_dataset( 'ml', data=levels[fidx])
  batch_item.create_dataset( 'datetime', data=time[0][bidx])
  batch_item.create_dataset( 'lat', data=lat[0][bidx])
  batch_item.create_dataset( 'lon', data=lon[0][bidx])

def write_batch_item_BERT_vertical(
  batch_item, fidx, bidx, data, levels, coords
):
  time, lat, lon = coords[:3]
  for vidx in range(len(levels[fidx])) :
    m_lvl = batch_item.require_group( f'ml={levels[fidx][vidx]}')
    m_lvl.create_dataset( 'data', data=data[vidx][bidx])
    m_lvl.create_dataset( 'ml', data=levels[fidx][vidx])
    m_lvl.create_dataset( 'datetime', data=time[fidx][bidx][vidx])
    m_lvl.create_dataset( 'lat', data=lat[fidx][bidx][vidx])
    m_lvl.create_dataset( 'lon', data=lon[fidx][bidx][vidx])

def get_zarr_store(name, model_id, epoch, zarr_store_type):
  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'
  return zarr_store_type( fname.format(name))

def write_zarr(
  name, fields, model_id, epoch, zarr_store_type, batch_idx, levels, coords, batch_item_write
):
  store = get_zarr_store(name, model_id, epoch, zarr_store_type)
  zarr_group = zarr.group(store=store)
  for fidx, field in enumerate(fields):
    name, data = field
    
    # for BERT: skip fields that were not predicted
    if (
      name in ["target", "pred", "ens"] and
      batch_item_write == write_batch_item_BERT_vertical and
      len(data[0]) == 0
    ):
      continue
      
    var_subgroup = zarr_group.require_group( f'{name}')
    batch_size = data.shape[0]
    for bidx in range(batch_size):
      sample_id = batch_idx * batch_size + bidx
      batch_item_subgroup = var_subgroup.create_group(f"sample={sample_id:05d}")
      batch_item_write(
        batch_item_subgroup, fidx, bidx, data, levels, coords
      )
      
  store.close()

def write_forecast(
  model_id, epoch, batch_idx, levels, sources, sources_coords, targets, targets_coords, preds, ensembles, zarr_store_type_str='ZipStore'
):
  ''' 
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''
  zarr_store_type = getattr( zarr, zarr_store_type_str)
  
  write_zarr(
    "source", sources, model_id, epoch, zarr_store_type, batch_idx, levels, sources_coords, write_batch_item_forecast
  )
  write_zarr(
    "target", targets, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_forecast
  )
  write_zarr(
    "pred", preds, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_forecast
  )
  write_zarr(
    "ens", ensembles, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_forecast
  )


def write_BERT(
  model_id, epoch, batch_idx, levels, sources, sources_coords, targets, targets_coords, preds, ensembles, zarr_store_type_str='ZipStore'
) :
  '''
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''
  zarr_store_type = getattr( zarr, zarr_store_type_str)
  
  write_zarr(
    "source", sources, model_id, epoch, zarr_store_type, batch_idx, levels, sources_coords, write_batch_item_BERT
  )
  write_zarr(
    "target", targets, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_BERT_vertical
  )
  write_zarr(
    "pred", preds, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_BERT_vertical
  )
  write_zarr(
    "ens", ensembles, model_id, epoch, zarr_store_type, batch_idx, levels, targets_coords, write_batch_item_BERT_vertical
  )


def write_attention(model_id, epoch, batch_idx, levels, attn, attn_coords, zarr_store_type = 'ZipStore' ) :

  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'
  zarr_store = getattr( zarr, zarr_store_type)

  store_attn = zarr_store( fname.format( 'attention'))
  exp_attn = zarr.group(store=store_attn)

  for fidx, atts_f in enumerate(attn) :
    ds_field = exp_attn.require_group( f'{atts_f[0]}')
    ds_field_b = ds_field.require_group( f'batch={batch_idx:05d}')
    for lidx, atts_f_l in enumerate(atts_f[1]) :  # layer in the network
      ds_f_l = ds_field_b.require_group( f'layer={lidx:05d}')
      ds_f_l.create_dataset( 'ml', data=levels[fidx])
      ds_f_l.create_dataset( 'datetime', data=attn_coords[0][fidx])
      ds_f_l.create_dataset( 'lat', data=attn_coords[1][fidx])
      ds_f_l.create_dataset( 'lon', data=attn_coords[2][fidx])
      ds_f_l_h = ds_f_l.require_group('heads')
      for hidx, atts_f_l_head in enumerate(atts_f_l) :  # number of attention head
        if atts_f_l_head != None :
          ds_f_l_h.create_dataset(f'{hidx}', data=atts_f_l_head.numpy() )
  store_attn.close()