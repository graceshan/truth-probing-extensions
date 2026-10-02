#!/usr/bin/env python3
"""CPU preparation only: inventory or bounded diagnostic timings, never production."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import checkpoint_r2_cpu_preflight_v1 as prep


def safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def worker(case, destination):
    import numpy as np
    import scipy, sklearn
    from threadpoolctl import threadpool_limits, threadpool_info
    from src import selection_repair_capacity_methods_v1 as methods
    from src import selection_repair_objectives as obj
    cfg = prep.config(ROOT)
    kind, model, method = case.split(':')
    settings = cfg['models'][model]
    rng = np.random.Generator(np.random.PCG64(cfg['benchmark']['seed']))
    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    overall = time.monotonic()
    if kind == 'historical':
        X, y, rows, input_pin = prep.historical_P15(ROOT, model)
        A = None
    else:
        n = 160 if kind == 'small' else 2778
        d = 24 if kind == 'small' else settings['width']
        n_a = 64 if kind == 'small' else 320
        X = rng.normal(size=(n, d))
        y = (X[:, 0] + .5 * X[:, 1] + rng.normal(size=n) > 0).astype(int)
        A = rng.normal(size=(n_a, d))
        ay = (A[:, 0] + .5 * A[:, 1] + rng.normal(size=n_a) > 0).astype(int)
        rows = []
        input_pin = dict(scope='synthetic_diagnostic_only', seed=cfg['benchmark']['seed'],
                         X_sha256=__import__('hashlib').sha256(X.tobytes()).hexdigest(),
                         y_sha256=__import__('hashlib').sha256(y.tobytes()).hexdigest(),
                         A_sha256=__import__('hashlib').sha256(A.tobytes()).hexdigest())
    with threadpool_limits(limits=1):
        threads = threadpool_info()
        started = time.monotonic()
        if method == 'synthetic_compound_objective':
            stats = obj.fit_p_preprocessing(X)
            p = obj.prepare_block(X, y, stats);a = obj.prepare_block(A, ay, stats)
            head = obj.fit_readout('compound', p, stats, adaptation=a)
            attempts = []
            for attempt in head.attempts:
                rec = asdict(attempt);rec.pop('initial_parameters');rec.pop('final_parameters');attempts.append(rec)
            details = dict(status=head.status, valid_for_scoring=head.converged, attempts=attempts,
                           preprocessing=stats.policy, objective='unchanged half P + half synthetic A + .001 norm squared')
            param_bytes = head.weights.nbytes
        else:
            params, details = methods.fit(method, X, y, rows, settings['C'], settings['canonical_layer'])
            param_bytes = sum(np.asarray(v).nbytes for v in params.values())
        elapsed = time.monotonic() - started
    record = dict(case=case, kind=kind, model=model, method=method, saved_layer=settings['canonical_layer'],
                  C=settings['C'] if method == 'l2_logistic' else None,
                  status='converged' if details['valid_for_scoring'] else 'flagged', details=details,
                  scope='historical_v4_P15_timing_not_production' if kind == 'historical' else 'synthetic_timing_not_research_fit',
                  fitting_rows=len(X),width=X.shape[1],dtype=str(X.dtype),adaptation_rows=0 if method != 'synthetic_compound_objective' else len(A),
                  preprocessing='P-only mean/population std with floor and constant mask' if method in ('r0','synthetic_compound_objective') else 'none; covariance-MM uses cohort within-class centering and absolute pinv cutoff 1e-3',
                  wall_fit_including_preprocessing_seconds=elapsed,wall_total_seconds=time.monotonic()-overall,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  baseline_process_peak_rss_bytes=base_rss, parameter_bytes=param_bytes,
                  memory_note='Darwin ru_maxrss is bytes; fresh process per case; peak includes input loading, float64 arrays and workspaces',
                  threads=threads, CPU_processes=1, input_binding=input_pin,params_saved=False,D_metrics_computed=False,
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                  preparation_base=prep.BASE,config=prep.identity(ROOT/prep.CONFIG),common_checkpoint_sha=None)
    Path(destination).write_text(json.dumps(safe(record),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(case,record['status'],round(elapsed,3),'s',flush=True)


def benchmark(output):
    prep.require(platform.system() == 'Darwin', 'timing session must remain on macOS')
    cfg = prep.config(ROOT)
    output = Path(output);prep.require(not output.exists(), 'refuse timing overwrite')
    output.mkdir(parents=True)
    cases = [f'small:qwen:{m}' for m in cfg['benchmark']['synthetic_representative']['methods']]
    cases += [f'synthetic:{model}:{method}' for model in cfg['benchmark']['synthetic_representative']['models']
              for method in cfg['benchmark']['synthetic_representative']['methods']]
    cases += [f'historical:{model}:{method}' for model in cfg['benchmark']['historical_cache']['models']
              for method in cfg['benchmark']['historical_cache']['methods']]
    started = time.monotonic();records = []
    for case in cases:
        path = output / (case.replace(':','_')+'.json')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        try:
            proc = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--case',case,'--output',str(path)],
                                  capture_output=True,text=True,env=env,timeout=cfg['benchmark']['timeout_seconds_per_case'])
            if proc.returncode:
                raise RuntimeError(proc.stderr[-6000:])
            record = json.loads(path.read_text())
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            record = dict(case=case,status='unavailable_or_failed',error=str(exc),production_started=False)
            path.write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
        print(case,record['status'],record.get('wall_fit_including_preprocessing_seconds'),flush=True)
    summary = dict(base=prep.BASE,common_checkpoint_sha=None,production_started=False,hostname=platform.node(),
                   elapsed_seconds=time.monotonic()-started,cases=records,config=prep.identity(ROOT/prep.CONFIG),
                   runner=prep.identity(Path(__file__)),loader=prep.identity(ROOT/'src/checkpoint_r2_cpu_preflight_v1.py'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['inventory','benchmark','worker'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case')
    args=parser.parse_args()
    if args.mode=='inventory':
        print(json.dumps(prep.write_inventory(ROOT,args.output),indent=2))
    elif args.mode=='benchmark':
        benchmark(args.output)
    else:
        cfg=prep.config(ROOT)
        kind,model,method=args.case.split(':')
        allowed_methods=cfg['benchmark']['historical_cache']['methods'] if kind=='historical' else cfg['benchmark']['synthetic_representative']['methods']
        prep.require(kind in ('small','synthetic','historical') and model in cfg['models'] and method in allowed_methods,'case not authorized')
        prep.require(not args.output.exists(), 'refuse timing checkpoint overwrite')
        worker(args.case,args.output)
