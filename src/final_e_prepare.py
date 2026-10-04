"""Deterministic preparation reconstruction; never opens E representations."""
import argparse
import fcntl
from pathlib import Path
from src.final_e_inventory import ROOT, Metadata, read
from src.final_e_registry import build
from src.final_e_reporting import schedules
from src.final_e_access import draft
from src.checkpoint_r2_fresh_inputs import require, hash_value
from src.checkpoint_r2_fresh_store import write_json, publish, pin


def forecast():
    prior=read(ROOT/'results/final_e_preparation_20261003/prior_raw_measurements.json')
    cfg=read(ROOT/'config/checkpoint_r2/full_raw_v1.json');metadata=Metadata()
    registry=read(ROOT/'config/final_e/head_registry_v1.json');carry=read(ROOT/'config/final_e/carryforward_v1.json')
    n=len(metadata.texts);models={}
    for model,spec in cfg['models'].items():
        measured=prior['models'][model];charge=prior['budget']['model_seconds'][model]
        shard_cost=measured['forward_write_verify_seconds'];fixed=charge-shard_cost
        point=fixed+shard_cost*n/measured['unique_texts']
        models[model]=dict(unique_texts=n,logical_bindings=19424,shards=(n+127)//128,fp16_payload_bytes=n*spec['layers']*spec['width']*2,
            measured_reference_texts=measured['unique_texts'],measured_reference_token_range=[measured['token_min'],measured['token_max']],
            measured_reference_model_seconds=charge,measured_reference_forward_write_verify_seconds=shard_cost,
            estimated_active_seconds=point,planning_2x_seconds=2*point,planning_4x_seconds=4*point,
            measured_reference_cuda_peak_bytes=measured['peak'],E_peak_memory_bytes=None,E_token_lengths=None,
            remaining_model_seconds=carry['remaining_seconds'][model],fits_under_remaining_at_4x=(4*point<carry['remaining_seconds'][model]),
            layer_verification_passes=len({h['saved_layer'] for h in registry['heads'].values() if h['model']==model}))
    cpu=read(ROOT/'results/final_e_preparation_20261003/D-reproduction.json')
    total=sum(m['fp16_payload_bytes'] for m in models.values())
    return dict(schema='final-e-resource-forecast-v1',status='estimate_not_execution_readiness',models=models,
        exact=dict(total_fp16_payload_bytes=total,two_payload_copies_bytes=2*total,logical_bindings_both_models=38848,
            unique_model_text_combinations=38380,scalar_scores_float64_bytes=73*(974+4*7200)*8,
            scalar_metric_draw_bytes=73*(7+4*28)*2000*8,all_layer_loading_pass_bytes=sum(m['fp16_payload_bytes']*m['layer_verification_passes'] for m in models.values())),
        cpu=dict(measured_D_reproduction_seconds=cpu['seconds'],measured_prior_full_bank_audit_seconds=150.73920570826158,
            estimated_report_seconds=cpu['seconds']*(7+4*28)/77,planning_end_to_end_seconds=[900,3600],worker_cap_seconds=10800,
            basis='Measured full D reconstruction plus expanded E wording endpoints; I/O/loader passes and E schedules need an unmeasured allowance.'),
        unmeasured=dict(E_token_lengths=True,E_inference_and_CUDA_peaks=True,E_token_audit_seconds=True,E_actual_metadata_storage=True,
            persistent_quota=True,transfer_bandwidth=True,CPU_E_loader_throughput=True),
        storage_policy='Future real two-copy both-model allocation/fsync/readback, temporary shard, serialized metadata, conservative other-token allowance and inherited safety reserve. Existing weights/output retained. No statvfs capacity guarantee.',
        forecast_rule='Fixed measured loading/other model overhead + measured per-text forward/write/verify cost. 2x/4x are planning scenarios, not confidence bounds or guarantees; E token distribution is unmeasured.',
        budget_policy='Keep 7200 active seconds/model cumulative, 1800 seconds/non-model setup, 10800 seconds/CPU job. Recheck all intervening charges at lock; no automatic budget reset.',
        execution_authorized=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    require(not out.resolve().is_relative_to(ROOT), 'preparation arrays require an external destination')
    out.mkdir(parents=True,exist_ok=False)
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=Metadata();reg=build(read(args.roots))
        require(reg==read(ROOT/'config/final_e/head_registry_v1.json'),'registry deterministic reconstruction')
        weights,schedule=schedules(m,read(ROOT/'config/final_e/preparation_v1.json')['bootstrap'])
        require(schedule==read(ROOT/'results/final_e_preparation_20261003/bootstrap.json'),'schedule deterministic reconstruction')
        import numpy as np
        publish(out/'E-bootstrap-weights.npz',lambda f:np.savez_compressed(f,**weights))
        write_json(out/'schedule-artifact.json',dict(artifact=pin(out/'E-bootstrap-weights.npz'),schedule_sha256=hash_value(schedule),E_scores_used=False))
        for name,value in [('inventory',m.receipt),('head-registry',reg),('bootstrap',schedule),('forecast',forecast()),('draft-lock',draft())]:
            write_json(out/(name+'.json'),value)
        print('Metadata/head registry/bootstrap reconstruction passed. E remains closed.')


if __name__=='__main__':main()
