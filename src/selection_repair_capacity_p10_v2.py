"""Versioned nested P10 fallback: exactly 36 new fits over the unchanged v4 contract."""
import json
import hashlib
import platform
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy, sklearn
from threadpoolctl import threadpool_limits
from src import selection_repair_capacity_pilot_v1 as prior
from src import selection_repair_capacity_validation_v1 as prior_validation
from src.clean_transfer_statistics import interval
s, methods = prior.s, prior.methods
CONFIG=Path('config/clean_protocol/capacity_pilot_p10_v2.json')
PACKAGE=Path('results/t2_capacity_pilot_v1_20261001')
SOURCES=['src/selection_repair_capacity_p10_v2.py',*prior.SOURCES]
REVIEWED_START='de776f1c86e4ddc464f79feac391704cbbaf0dc0'


def file_identity(path):
    path=Path(path);h=hashlib.sha256();before=path.stat()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    after=path.stat()
    s.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'unstable input file')
    return dict(bytes=after.st_size,sha256=h.hexdigest())


def verify_config(root):
    root=Path(root);config=json.loads((root/CONFIG).read_text())
    for pin in config['bindings'].values():
        s.require(file_identity(root/pin['path'])=={k:pin[k] for k in ('bytes','sha256')},'stale P10 authorization binding: '+pin['path'])
    old=json.loads((root/prior.CONFIG).read_text())
    s.require(config['reviewed_start']==REVIEWED_START and config['correction_commit']==s.CORRECTION_COMMIT
              and config['cohorts']==['P10'] and config['additional_configurations']==36
              and config['P_A_frozen'] is False,'P10 authorization scope')
    for key in ('models','methods','endpoints','bootstrap','topics','scopes','review_drop_thresholds',
                'raw_lr','difference_of_means','mass_mean_covariance','ttpd','r0','fit_policy'):
        s.require(config[key]==old[key],'P10 numerical/endpoint policy differs: '+key)
    s.require(config['adoption_sha256']==old['adoption_sha256'],'P10 representation predecessor changed')
    return config


def membership(root):
    root=Path(root);members,samples=prior.cohorts(root);base=root/s.projection.PROJECTION
    a10_data=(base/'A10_proposal.json').read_bytes();a15_data=(base/'A15_proposal.json').read_bytes()
    for n,data in [(10,a10_data),(15,a15_data)]:
        s.require(data==(root/s.projection.prior.PACKAGE/f'A{n}_proposal.json').read_bytes(),'A proposal changed')
    a10=json.loads(a10_data)['completed_pairs'];a15=json.loads(a15_data)['completed_pairs']
    keys10={r['person_key'] for r in a10};keys15={r['person_key'] for r in a15}
    s.require(len(a10)==len(keys10)==50 and len(a15)==len(keys15)==75 and keys10<keys15,'nested A identity count')
    s.require(all(r in a15 for r in a10),'A10 fact records differ from A15')
    for topic in {r['topic'] for r in a15}:
        s.require(sum(r['topic']==topic for r in a10)==10 and sum(r['topic']==topic for r in a15)==15,'per-topic A allocation')
    rows=s.parse_rows((base/'P10_admitted_membership.csv').read_bytes())
    expected=[r for r in members['full'] if r['person_key'] not in keys10]
    s.require(rows==expected and len(rows)==2864,'P10 must equal corrected full minus every A10 source variant')
    ids=lambda rows:[r['source_row_id'] for r in rows]
    s.require(len(set(ids(rows)))==len(rows) and set(ids(members['P15']))<set(ids(rows)),'P15 not nested in P10')
    s.require(not keys10 & {r['person_key'] for r in rows} and all(r[s.STATUS]=='admitted' for r in rows),'A10/restricted source leak')
    manifest=s.parse_rows((root/s.projection.OVERLAY/'row_manifest.csv').read_bytes())
    excluded={r['source_row_id'] for r in manifest if r['split']=='train' and r['person_key'] in keys10}
    s.require(not excluded & set(ids(rows)),'A10 source variant leaked')
    sample=s.projection.prior.balanced(rows)
    saved=s.parse_rows((base/'P10_balanced_exposure.csv').read_bytes())
    s.require(sample==saved and len(sample)==800,'exact P10 sampler identity mismatch')
    for c in prior.counts(sample)[1:]:
        s.require(c['rows']==160 and c['labels']=={'0':80,'1':80},'P10 balanced per-topic exposure')
    before=set(ids(samples['P15']));after=set(ids(sample))
    report=dict(full_rows=len(members['full']),P15_rows=len(members['P15']),P10_rows=len(rows),
        A10_rows=50,A15_rows=75,A10_nested=True,proposal_bytes_unchanged=True,P15_subset_P10=True,
        all_A10_source_variants_excluded=True,A10_source_variant_count=len(excluded),
        A10_identity=s.identity(a10_data),A15_identity=s.identity(a15_data),
        membership_sha256=s.sha(s.canonical(rows)),ordered_ids_sha256=s.sha(s.canonical(ids(rows))),
        sampled_sha256=s.sha(s.canonical(sample)),ordered_sample_ids_sha256=s.sha(s.canonical(ids(sample))),
        sample_rows=len(sample),sample_counts=prior.counts(sample),P10_counts=prior.counts(rows),
        sample_comparison_to_P15=dict(shared=len(before&after),removed=sorted(before-after),added=sorted(after-before),
                                     P15_sample_subset_P10=before<=after),P_A_frozen=False)
    return rows,sample,report


def configurations(config):
    return [dict(model=model,layer=layer,method=method,cohort='P10',C=settings['C'])
            for model,settings in config['models'].items() for layer in settings['layers'] for method in config['methods']]


def locate_prior(root):
    """Every predecessor location comes from its completed committed manifests."""
    root=Path(root);external=json.loads((root/PACKAGE/'external_artifacts.json').read_text())
    old=Path(external['pilot_root']);prepared=Path(external['prepared_root']);exports=Path(external['verified_export_root'])
    for name in ('completion.json','binding.json','fits.json','metrics.json','bootstrap_binding.json','threshold_triggers.json','layer_bindings.json'):
        s.require((old/name).read_bytes()==(root/PACKAGE/name).read_bytes(),'predecessor manifest mismatch: '+name)
    for name,pin in external['artifacts'].items():s.require(file_identity(name)==pin,'predecessor export changed: '+name)
    inventory=json.loads((prepared/'input_inventory.json').read_text())
    artifacts=Path(inventory['physical_export_root'])
    validation=prior_validation.validate(root,prepared,old)
    completion=json.loads((old/'completion.json').read_text())
    s.require(completion['status']=='complete' and completion['unique_configurations']==72 and not completion['failures'],'predecessor pilot incomplete')
    runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__)
    s.require(runtime==completion['runtime'],'numerical environment changed')
    return old,prepared,exports,artifacts,validation


def metric_key(row):
    return tuple(row[k] for k in ('model','layer','method','metric','scope','topic'))


def require_draw_alignment(actual,expected):
    s.require(np.array_equal(actual,expected,equal_nan=True),'paired draws differ from predecessor')


def classify_trigger(old_delta,new_delta,threshold):
    """Direction and threshold persistence are separate from uncertainty."""
    if new_delta is None:return dict(trajectory='not_computed',threshold_status='not_computed')
    trajectory='improved' if new_delta>old_delta else 'worsened' if new_delta<old_delta else 'persistent'
    return dict(trajectory=trajectory,threshold_status='persistent' if new_delta < -threshold else 'cleared')


def trigger_followup(old_metrics,records):
    lookup={(r['comparison'],metric_key(r)):r for r in records}
    original=[];new=[]
    for old in old_metrics:
        key=metric_key(old);current=lookup.get(('P10_minus_full',key))
        if old['triggered']:
            delta=current['delta'] if current else None
            original.append(dict(identity={k:old[k] for k in ('model','layer','method','metric','scope','topic')},
                original_P15_minus_full=old['withholding_delta'],P10_minus_full=delta,
                P10_minus_P15=None if delta is None else delta-old['withholding_delta'],
                original_paired_ci=old['paired_delta_ci'],P10_full_paired_ci=current['paired_delta_ci'] if current else None,
                P10_P15_paired_ci=lookup[('P10_minus_P15',key)]['paired_delta_ci'] if current else None,
                threshold=old['review_drop_threshold'],**classify_trigger(old['withholding_delta'],delta,old['review_drop_threshold'])))
        elif current and current['triggered']:new.append(current)
    s.require(len(original)==14,'original trigger set changed')
    return dict(original_14=original,new_P10_vs_full_triggers=new,
                P10_vs_P15_triggers=[r for r in records if r['comparison']=='P10_minus_P15' and r['triggered']],
                note='Point changes and strict threshold crossings are separate from pointwise paired uncertainty; comparisons are dependent and not simultaneous tests.')


def evaluate(bundle,config,old,output,reports):
    atom=bundle.atomic;raw=bundle.raw
    atom_w=prior.atomic_weights(atom,np.ones(len(atom),bool),config['bootstrap'])
    comp_w,schedule=prior.compound_weights(raw,np.ones(len(raw),bool),config['bootstrap'])
    bootstrap=dict(compound_schedule_sha256=schedule,atomic_weight_sha256=s.sha(atom_w.tobytes()))
    s.require(bootstrap==json.loads((old/'bootstrap_binding.json').read_text()),'bootstrap entity schedule mismatch')
    (output/'bootstrap_binding.json').write_bytes(s.json_bytes(bootstrap))
    previous=json.loads((old/'metrics.json').read_text());prior_index={metric_key(r):i for i,r in enumerate(previous)}
    s.require(len(prior_index)==len(previous)==1260,'predecessor metric identities')
    with np.load(old/'scores'/'paired_draws.npz',allow_pickle=False) as archive:old_draws={k:archive[k] for k in archive.files}
    reports={r['id']:r for r in reports};records=[];draws=[];missing=[];aligned=0
    for model,settings in config['models'].items():
        for layer in settings['layers']:
            for method in config['methods']:
                name=f'{model}_L{layer}_{method}_P10'
                if not reports[name]['valid_for_scoring']:
                    missing.append(dict(model=model,layer=layer,method=method,reason='flagged/failed P10 fit'));continue
                with np.load(output/'scores'/(name+'.npz'),allow_pickle=False) as a:p10={k:a[k] for k in a.files}
                per_baseline={}
                for baseline in ('full','P15'):
                    with np.load(old/'scores'/f'{model}_L{layer}_{method}_{baseline}.npz',allow_pickle=False) as a:base={k:a[k] for k in a.files}
                    for definition in config['endpoints']:
                        if definition['condition']=='atomic_D':
                            key='atomic';labels=np.array([int(r['label']) for r in atom]);mask=np.ones(len(atom),bool);topics=np.array([r['topic'] for r in atom]);weights=atom_w
                        else:
                            key='bare';labels=raw.cell.isin(definition['positive']).to_numpy(int)
                            mask=((raw.operator==definition['operator'])&raw.cell.isin(definition['positive']+definition['negative'])).to_numpy();topics=raw.topic.to_numpy();weights=comp_w
                        found,pairs=prior.compare_endpoint(base[key],p10[key],labels,mask,np.ones(len(labels),bool),topics,weights,definition,model,config['bootstrap'],config['topics'])
                        for r,(baseline_draw,p10_draw) in zip(found,pairs):
                            r.update(layer=layer,method=method);i=prior_index[metric_key(r)];saved=previous[i]
                            s.require(r['retained_rows']==saved['retained_rows'] and r['historical_retained_auroc']==saved[baseline+'_auroc'],'baseline point/coverage alignment')
                            require_draw_alignment(baseline_draw,old_draws[baseline][:,i])
                            if baseline=='full':per_baseline[i]=p10_draw
                            else:s.require(np.array_equal(p10_draw,per_baseline[i],equal_nan=True),'P10 draws differ across comparisons')
                            aligned+=1
                            record=dict(model=model,layer=layer,method=method,comparison='P10_minus_'+baseline,baseline=baseline,
                                condition=r['condition'],metric=r['metric'],scope=r['scope'],topic=r['topic'],status=r['status'],
                                baseline_auroc=r['historical_retained_auroc'],P10_auroc=r['corrected_retained_auroc'],delta=r['training_delta'],
                                rows=r['retained_rows'],baseline_ci=r['historical_ci'],P10_ci=r['corrected_ci'],paired_delta_ci=r['paired_delta_ci'],
                                prior_metric_index=i,review_drop_threshold=config['review_drop_thresholds'].get(r['metric']))
                            threshold=record['review_drop_threshold'];record['triggered']=threshold is not None and record['delta'] is not None and record['delta'] < -threshold
                            if record['status']!='computed' or record['paired_delta_ci']['ci_status']!='ok':missing.append(dict(configuration=name,comparison=record['comparison'],metric=record['metric'],scope=record['scope'],topic=record['topic'],reason='endpoint or uncertainty undefined'))
                            records.append(record);draws.append((baseline_draw,p10_draw))
                print(f'Evaluated both comparisons: {model} L{layer} {method}',flush=True)
    return records,draws,missing,dict(aligned_baseline_summary_draw_vectors=aligned,identical_D_rows=True,identical_entity_schedule=True,predecessor_draws_bitwise_equal=True)


def run(root,payload,output,resume=False):
    root,output=Path(root),Path(output)
    s.require(platform.system()=='Darwin','P10 fit must stay on macOS')
    s.require(not output.resolve().is_relative_to(root.resolve()),'external output required')
    config=verify_config(root);rows,sample,exposure=membership(root)
    old,prepared,exports,artifacts,old_validation=locate_prior(root)
    bundle=prior.PilotInputs(root,payload,prepared,artifacts,exports)
    s.require((old/'atomic_D_rows.csv').read_bytes()==s.csv_bytes(bundle.atomic)
              and (old/'bare_D_rows.csv').read_bytes()==bundle.raw.to_csv(index=False,lineterminator='\n').encode(),'D ordered identity/label mismatch')
    configs=configurations(config);s.require(len(configs)==36,'exact P10 grid')
    binding=dict(config=file_identity(root/CONFIG),prior_completion=file_identity(old/'completion.json'),prior_root=str(old),
        prepared_root=str(prepared),exports_root=str(exports),physical_artifact_root=str(artifacts),payload_root=str(Path(payload).resolve()),
        preparation=file_identity(prepared/'preparation_receipt.json'),membership=s.sha(s.canonical(exposure)),
        source_sha256={name:s.sha((root/name).read_bytes()) for name in SOURCES})
    if output.exists():
        s.require(resume and (output/'binding.json').read_bytes()==s.json_bytes(binding),'incompatible P10 resume; no overwrite')
        s.require(not (output/'completion.json').exists(),'completed P10 output immutable')
    else:
        output.mkdir(parents=True);(output/'fits').mkdir();(output/'scores').mkdir()
        (output/'binding.json').write_bytes(s.json_bytes(binding));(output/'predeclared_config.json').write_bytes(s.json_bytes(config))
        (output/'membership.json').write_bytes(s.json_bytes(exposure));(output/'P10_membership.csv').write_bytes(s.csv_bytes(rows))
        (output/'P10_balanced.csv').write_bytes(s.csv_bytes(sample))
        (output/'predecessor_validation.json').write_bytes(s.json_bytes(old_validation))
        for name in ('atomic_D_rows.csv','bare_D_rows.csv'):(output/name).write_bytes((old/name).read_bytes())
    start=time.monotonic();reports=[];replay=[]
    old_fits={r['id']:r for r in json.loads((old/'fits.json').read_text())}
    layer_bindings=json.loads((old/'layer_bindings.json').read_text())
    with threadpool_limits(limits=1):
        for model,settings in config['models'].items():
            for layer in settings['layers']:
                arrays,layer_pin=bundle.layer(model,layer)
                s.require(layer_pin==layer_bindings[model+'_L'+str(layer)],'layer/source/float64 identity differs')
                pos={r['source_row_id']:i for i,r in enumerate(bundle.memberships['full'])}
                # Reproduce every full/P15 score and fitting identity without fitting again.
                for method in config['methods']:
                    for cohort in ('full','P15'):
                        name=f'{model}_L{layer}_{method}_{cohort}';saved=old_fits[name]
                        fit_rows=(bundle.balanced if method in ('burger_t_g','ttpd') else bundle.memberships)[cohort]
                        X=arrays['train'][[pos[r['source_row_id']] for r in fit_rows]]
                        s.require(s.sha(X.tobytes())==saved['fitting_identity']['feature_float64_sha256'],'predecessor fit-state identity mismatch')
                        with np.load(old/'fits'/name/'parameters.npz',allow_pickle=False) as a:params={k:a[k] for k in a.files}
                        with np.load(old/'scores'/(name+'.npz'),allow_pickle=False) as a:
                            for endpoint in ('atomic','bare'):
                                actual=methods.score(method,params,arrays[endpoint]);s.require(np.array_equal(actual,a[endpoint]),'predecessor saved score replay mismatch')
                        replay.append(dict(configuration=name,fit_identity_verified=True,score_replay_bitwise_equal=True,parameters=saved['parameters']))
                        del X
                for method in config['methods']:
                    name=f'{model}_L{layer}_{method}_P10';configuration=dict(model=model,layer=layer,method=method,cohort='P10',C=settings['C'])
                    fit_rows=sample if method in ('burger_t_g','ttpd') else rows
                    idx=np.array([pos[r['source_row_id']] for r in fit_rows]);X=arrays['train'][idx];y=np.array([int(r['label']) for r in fit_rows])
                    fit_pin=dict(membership_sha256=s.sha(s.canonical(fit_rows)),ordered_ids_sha256=s.sha(s.canonical([r['source_row_id'] for r in fit_rows])),
                        feature_float64_sha256=s.sha(X.tobytes()),labels_sha256=s.sha(y.astype('<i8').tobytes()),layer_binding=layer_pin['train'])
                    directory=output/'fits'/name
                    if directory.exists():
                        report=json.loads((directory/'fit.json').read_text())
                        s.require(report['configuration']==configuration and report['fitting_identity']==fit_pin,'P10 checkpoint identity mismatch')
                        if 'parameters' in report:s.require(file_identity(directory/'parameters.npz')==report['parameters'],'P10 checkpoint parameter change')
                    else:
                        started=time.monotonic();params=None
                        details=dict(fitting_identity=fit_pin,fit_rows=len(fit_rows),exposure=prior.counts(fit_rows),A_exposed_diagnostic=False,
                            primary_bank_eligible=False,P_A_frozen=False,preprocessing_fitting_cohort='P10',reused_canonical=False)
                        try:
                            params,diagnostics=methods.fit(method,X,y,fit_rows,settings['C'],layer)
                            finite=all(np.isfinite(v).all() for v in params.values() if np.asarray(v).dtype.kind not in 'US')
                            details.update(diagnostics,finite_parameters=bool(finite));details['valid_for_scoring']=bool(details['valid_for_scoring'] and finite)
                            details['status']='converged' if details['valid_for_scoring'] else 'flagged'
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                            details.update(status='failed',valid_for_scoring=False,error=repr(exc),traceback=traceback.format_exc())
                        details['elapsed_seconds']=time.monotonic()-started
                        prior.checkpoint(directory,configuration,method,params,details)
                        report=json.loads((directory/'fit.json').read_text())
                    report['id']=name;reports.append(report)
                    print(name+' '+report['status']+' '+str(round(report['elapsed_seconds'],2))+'s',flush=True)
                    if report['valid_for_scoring']:
                        with np.load(directory/'parameters.npz',allow_pickle=False) as a:params={k:a[k] for k in a.files}
                        scores={k:methods.score(method,params,arrays[k]) for k in ('atomic','bare')};path=output/'scores'/(name+'.npz')
                        if path.exists():
                            with np.load(path,allow_pickle=False) as a:
                                for k in scores:s.require(np.array_equal(a[k],scores[k]),'P10 checkpoint score mismatch')
                        else:np.savez_compressed(path,**scores)
                    del X
                del arrays
        (output/'fits.json').write_bytes(s.json_bytes(reports));(output/'predecessor_replay.json').write_bytes(s.json_bytes(replay))
        records,draws,missing,alignment=evaluate(bundle,config,old,output,reports)
    elapsed=time.monotonic()-start
    (output/'metrics.json').write_bytes(s.json_bytes(records))
    flat=[]
    for record in records:
        row={k:v for k,v in record.items() if not isinstance(v,dict)};row.update(record['paired_delta_ci']);flat.append(row)
    if flat:(output/'metrics.csv').write_bytes(s.csv_bytes(flat))
    np.savez_compressed(output/'scores'/'paired_draws.npz',baseline=np.array([a for a,b in draws]).T,P10=np.array([b for a,b in draws]).T)
    (output/'alignment.json').write_bytes(s.json_bytes(alignment))
    followup=trigger_followup(json.loads((old/'metrics.json').read_text()),records)
    (output/'trigger_followup.json').write_bytes(s.json_bytes(followup))
    verify_config(root);bundle.bundle.unchanged();prior.staging.verify(root,prepared,exports)
    for name,pin in json.loads((old/'completion.json').read_text())['artifacts'].items():s.require(file_identity(old/name)==pin,'predecessor artifact changed during P10')
    s.require(binding['source_sha256']=={name:s.sha((root/name).read_bytes()) for name in SOURCES},'P10 source changed during run')
    receipt=dict(status='complete' if not missing and all(r['valid_for_scoring'] for r in reports) else 'completed_with_failures',
        correction_commit=s.CORRECTION_COMMIT,completed_utc=datetime.now(timezone.utc).isoformat(),local_platform=platform.system(),hostname=platform.node(),
        unique_new_configurations=len(reports),prior_configurations=72,cumulative_unique_configurations=72+len(reports),new_configuration_fits=len(reports),
        failures=[r['id'] for r in reports if not r['valid_for_scoring']],missing_comparisons=missing,endpoint_summaries=len(records),
        elapsed_seconds=elapsed,fit_seconds=sum(r['elapsed_seconds'] for r in reports),cpu_concurrency=1,blas_threads=1,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__),
        P_A_frozen=False,production_bank_fits=0,repair_fits=0,new_transfers=0,model_extractions=0,final_test_access=False,
        artifacts={str(p.relative_to(output)):file_identity(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'completion.json').write_bytes(s.json_bytes(receipt));return receipt
