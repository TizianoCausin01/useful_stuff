import os, yaml, sys
import numpy as np

sys.path.append("..")
from general_utils.utils import print_wise, TimeSeries, dRSA


def across_areas_dRSA(paths: dict[str: str], rank: int, target_areas: tuple[str], raster: 'TimeSeries', brain_areas_obj: 'BrainAreas', cfg):
    outfn = f"{paths['livingstone_lab']}/tiziano/results/{cfg.monkey_name}_{cfg.date}_{target_areas[0]}-{target_areas[1]}_images_{cfg.RDM_metric}_{cfg.new_fs}Hz.npz"
    if os.path.exists(outfn):
        print_wise(f"File already exists at {outfn}", rank=rank)
    else:
        print_wise(f"Start computing dRSA across {target_areas[0]} and {target_areas[1]}", rank=rank)
        brain_area_signal = brain_areas_obj.slice_brain_area(raster, target_areas[0])
        brain_area_model = brain_areas_obj.slice_brain_area(raster, target_areas[1])
        drsa_obj = dRSA(cfg.RDM_metric, RSA_metric=cfg.RSA_metric)    
        drsa_obj.compute_both_RDM_timeseries(brain_area_signal, brain_area_model)
        drsa_mat = drsa_obj.compute_dRSA()
        np.savez_compressed(outfn, data=drsa_mat)
        print_wise(f"Computed dRSA across {target_areas[0]} and {target_areas[1]} \nat {outfn}", rank=rank)
    # end if os.path.exists(outfn):
# EOF
