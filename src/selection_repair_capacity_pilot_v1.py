"""Bounded v4 full/P15 capacity pilot: 72 fixed configurations, no bank or repair."""
import io
import json
import platform
import time
import traceback
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy, sklearn
from threadpoolctl import threadpool_limits,threadpool_info
from src import selection_repair_sensitivity_v4 as s
from src import selection_repair_capacity_methods_v1 as methods
from src import selection_repair_capacity_staging_v1 as staging
from src.selection_repair_canonical_sensitivity_v4 import BoundInputs
from src.selection_repair_canonical_sensitivity import atomic_weights,compound_weights,compare_endpoint
from src.selection_repair_cache_export_remote import DIRECTORIES

CONFIG=Path('config/clean_protocol/capacity_pilot_v1.json')
SOURCES=['src/selection_repair_capacity_pilot_v1.py','src/selection_repair_capacity_methods_v1.py',
         'src/selection_repair_capacity_staging_v1.py','src/selection_repair_capacity_export_v2.py',
         'src/selection_repair_objectives.py','src/atomic_probe_methods.py','src/clean_atomic_probes.py',
         'src/selection_repair_canonical_sensitivity.py','src/clean_transfer_statistics.py']


def counts(rows):
    result=[]
    for topic in ['all']+sorted({r['topic'] for r in rows}):
        group=[r for r in rows if topic=='all' or r['topic']==topic]
        result.append(dict(topic=topic,rows=len(group),entities=len({r['entity_id'] for r in group}),
            person_keys=len({r['person_key'] for r in group}),labels=dict(Counter(r['label'] for r in group)),forms=dict(Counter(r['form'] for r in group))))
    return result


def cohorts(root):
    root=Path(root);base=root/s.projection.PROJECTION
    names={'full':'corrected_outer_training','P15':'P15_admitted_membership'}
    memberships={k:s.parse_rows((base/(v+'.csv')).read_bytes()) for k,v in names.items()}
    a=json.loads((base/'A15_proposal.json').read_text())['completed_pairs'];keys={r['person_key'] for r in a}
    s.require(len(keys)==75,'A15 identity count')
    ids=lambda rows:[r['source_row_id'] for r in rows]
    s.require(ids(memberships['P15'])==ids([r for r in memberships['full'] if r['person_key'] not in keys]),'P15 source variant exclusion')
    balanced={}
    for cohort,rows in memberships.items():
        s.require(len(rows)==(3040 if cohort=='full' else 2778) and len(set(ids(rows)))==len(rows),'cohort identity count')
        s.require(all(r[s.STATUS]=='admitted' and r['split']=='train' for r in rows) and not set(ids(rows))&s.QUARANTINED,'restricted cohort')
        sampled=s.projection.prior.balanced(rows)
        file='corrected_outer_training_balanced.csv' if cohort=='full' else 'P15_balanced_exposure.csv'
        s.require(sampled==s.parse_rows((base/file).read_bytes()),'exact ordered sampler mismatch')
        s.require(len(sampled)==(1000 if cohort=='full' else 700),'balanced exposure count')
        if cohort=='P15':s.require(not keys & {r['person_key'] for r in sampled+rows},'A source variant leaked')
        balanced[cohort]=sampled
    return memberships,balanced


class PilotInputs:
    def __init__(self,root,payload,prepared,artifacts,exports):
        self.bundle=BoundInputs(root,payload,prepared,artifacts)
        self.exports=Path(exports);staging.verify(root,prepared,exports)
        expected,bindings=staging.make_request(root,payload,prepared,artifacts)
        s.require((self.exports/'export_request.json').read_bytes()==s.json_bytes(expected)
                  and (self.exports/'row_bindings.json').read_bytes()==s.json_bytes(bindings),'pilot export identity recomputation mismatch')
        self.request=expected;self.bindings=bindings
        self.remote=json.loads((self.exports/'export_receipt.json').read_text())['caches']
        self.atomic=[r for r in s.parse_rows((Path(prepared)/'atomic_D_eligibility.csv').read_bytes()) if r['tc_current_eligible']=='True']
        raw=s.parse_rows(self.bundle.read(s.RAW+'metadata.csv'))
        pairs=s.parse_rows((Path(prepared)/'development_pair_eligibility.csv').read_bytes())
        allowed={r['pair_id'] for r in pairs if r['tc_current_eligible']=='True'}
        self.raw=pd.DataFrame([r for r in raw if r['pair_id'] in allowed])
        self.raw['cell']=np.where(self.raw.canonical_truth_a=='True','T','F')+np.where(self.raw.canonical_truth_b=='True','T','F')
        s.require(len(self.atomic)==1012 and len(self.raw)==483*16,'pilot D coverage')
        self.memberships,self.balanced=cohorts(root)
        self.arrays={}

    def layer(self,model,layer):
        result={};receipts={}
        for endpoint,key in [('train',model+'_train'),('atomic',model+'_validation'),('bare','qwen_raw' if model=='qwen' else 'llama_transfer')]:
            desired=self.bindings[key]['original_row_indices']
            if layer==self.bundle.config['models'][model]['layer']:
                original=self.bundle.request[key]['row_indices'];lookup={r:i for i,r in enumerate(original)}
                s.require(all(i in lookup for i in desired),'missing original cached export row')
                array=np.asarray(self.bundle.array(key)[[lookup[i] for i in desired]],dtype=np.float64)
                receipts[endpoint]=dict(source=self.bundle.remote[key]['source'],export=self.bundle.remote[key]['export'])
            else:
                job=key+'_L'+str(layer)
                array=np.asarray(np.load(self.exports/(job+'.npy'),mmap_mode='r',allow_pickle=False),dtype=np.float64)
                receipts[endpoint]=dict(source=self.remote[job]['source'],export=self.remote[job]['export'])
            s.require(array.shape==(len(desired),self.bundle.config['models'][model]['width']) and np.isfinite(array).all(),'layer shape/finite check')
            if endpoint=='bare':
                names=[r['sidecar']['example_id'] for r in self.bindings[key]['identities']]
                pos={name:i for i,name in enumerate(names)}
                s.require(set(names)==set(self.raw.example_id) and len(names)==len(pos),'bare eval identity mismatch')
                array=array[[pos[e] for e in self.raw.example_id]]
            result[endpoint]=array
            receipts[endpoint].update(original_row_indices_sha256=s.sha(s.canonical(desired)),
                row_binding_sha256=s.sha(s.canonical(self.bindings[key]['identities'])),
                materialized_float64_sha256=s.sha(array.tobytes()))
        train_ids=[r['source_row_id'] for r in self.bindings[model+'_train']['identities']]
        s.require(train_ids==[r['source_row_id'] for r in self.memberships['full']],'training layer identity order')
        return result,receipts


def checkpoint(directory,identity,method,params,details):
    directory.mkdir()
    if params is not None:
        np.savez_compressed(directory/'parameters.npz',**params)
        details['parameters']=s.identity((directory/'parameters.npz').read_bytes())
    (directory/'fit.json').write_bytes(s.json_bytes(dict(configuration=identity,**details)))


def configurations(config):
    return [dict(model=model,layer=layer,method=method,cohort=cohort,C=settings['C'])
            for model,settings in config['models'].items() for layer in settings['layers']
            for method in config['methods'] for cohort in ('full','P15')]


def run(root,payload,prepared,artifacts,exports,canonical_results,output,resume=False):
    root,prepared,output,canonical_results=map(Path,(root,prepared,output,canonical_results))
    s.require(platform.system()=='Darwin','pilot must remain on macOS')
    s.require(not output.resolve().is_relative_to(root.resolve()),'large artifacts outside checkout')
    config=json.loads((root/CONFIG).read_text());canonical_config,_=s.verify_preparation(root,prepared)
    s.require(config['correction_commit']==s.CORRECTION_COMMIT and config['adoption_sha256']==canonical_config['adoption_sha256'],'stale pilot adoption')
    s.require(config['methods']==list(methods.METHODS) and {k:v['layers'] for k,v in config['models'].items()}==staging.LAYERS,'pilot configurations changed')
    s.require({m:v['C'] for m,v in config['models'].items()}=={'qwen':1.0,'llama':10.0},'pinned pilot C changed')
    s.require(config['bootstrap']==canonical_config['bootstrap'] and config['endpoints']==canonical_config['endpoints'][:5],'pilot endpoint/uncertainty changed')
    configs=configurations(config);s.require(len(configs)==72,'expected 72 unique configurations')
    bundle=PilotInputs(root,payload,prepared,artifacts,exports)
    canonical_receipt=json.loads((canonical_results/'sensitivity_receipt.json').read_text())
    s.require(canonical_receipt['preparation_receipt']==s.identity((prepared/'preparation_receipt.json').read_bytes()) and canonical_receipt['correction_commit']==s.CORRECTION_COMMIT,'canonical reuse correction mismatch')
    canonical_fits=json.loads((canonical_results/'fits.json').read_text())
    for name,pin in canonical_receipt['artifacts'].items():s.require(s.identity((canonical_results/name).read_bytes())==pin,'canonical artifact changed')
    binding=dict(config=s.identity((root/CONFIG).read_bytes()),preparation=s.identity((prepared/'preparation_receipt.json').read_bytes()),
        export_receipt=s.identity((Path(exports)/'export_receipt.json').read_bytes()),canonical_receipt=s.identity((canonical_results/'sensitivity_receipt.json').read_bytes()),
        source_sha256={name:s.sha((root/name).read_bytes()) for name in SOURCES})
    if output.exists():
        s.require(resume and (output/'binding.json').read_bytes()==s.json_bytes(binding),'refuse overwrite or incompatible resume')
        s.require(not (output/'completion.json').exists(),'completed pilot is immutable')
    else:
        output.mkdir(parents=True);(output/'fits').mkdir();(output/'scores').mkdir();(output/'membership').mkdir()
        (output/'binding.json').write_bytes(s.json_bytes(binding))
        (output/'predeclared_endpoints.json').write_bytes(s.json_bytes(config))
        (output/'atomic_D_rows.csv').write_bytes(s.csv_bytes(bundle.atomic))
        (output/'bare_D_rows.csv').write_bytes(bundle.raw.to_csv(index=False,lineterminator='\n').encode())
        for cohort in ('full','P15'):
            for suffix,rows in [('',bundle.memberships[cohort]),('_balanced',bundle.balanced[cohort])]:
                (output/'membership'/(cohort+suffix+'.csv')).write_bytes(s.csv_bytes(rows))
    start=time.monotonic();reports=[];layer_bindings={}
    with threadpool_limits(limits=1):
        for model,settings in config['models'].items():
            for layer in settings['layers']:
                arrays,layer_pin=bundle.layer(model,layer);layer_bindings[model+'_L'+str(layer)]=layer_pin
                pos={r['source_row_id']:i for i,r in enumerate(bundle.memberships['full'])}
                for method in config['methods']:
                    for cohort in ('full','P15'):
                        name=f'{model}_L{layer}_{method}_{cohort}'
                        identity=dict(model=model,layer=layer,method=method,cohort=cohort,C=settings['C'])
                        directory=output/'fits'/name
                        rows=(bundle.balanced if method in ('burger_t_g','ttpd') else bundle.memberships)[cohort]
                        idx=np.array([pos[r['source_row_id']] for r in rows]);X=arrays['train'][idx];y=np.array([int(r['label']) for r in rows])
                        fit_pin=dict(membership_sha256=s.sha(s.canonical(rows)),ordered_ids_sha256=s.sha(s.canonical([r['source_row_id'] for r in rows])),
                                     feature_float64_sha256=s.sha(X.tobytes()),labels_sha256=s.sha(y.astype('<i8').tobytes()),layer_binding=layer_pin['train'])
                        if directory.exists():
                            report=json.loads((directory/'fit.json').read_text())
                            s.require(report['configuration']==identity and report['fitting_identity']==fit_pin,'stale fit checkpoint')
                            if 'parameters' in report:s.require(s.identity((directory/'parameters.npz').read_bytes())==report['parameters'],'checkpoint parameters changed')
                        else:
                            started=time.monotonic();params=None
                            details=dict(fitting_identity=fit_pin,exposure=counts(rows),fit_rows=len(rows),
                                A_exposed_diagnostic=cohort=='full',primary_bank_eligible=False,preprocessing_fitting_cohort=cohort,reused_canonical=False)
                            try:
                                if method=='l2_logistic' and cohort=='full' and layer==canonical_config['models'][model]['layer']:
                                    f=canonical_fits[model]
                                    mapping=(prepared/(model+'_train.mapping_v4.csv')).read_bytes()
                                    s.require(f['training_membership']==s.identity(mapping) and f['training_rows']==len(rows) and f['C']==settings['C']
                                              and f['saved_layer_index']==layer and f['preprocessing']=='none' and f['valid_for_scoring'],'canonical fitting identity mismatch')
                                    s.require(f['source_export']==layer_pin['train']['export'],'canonical source export mismatch')
                                    from src.selection_repair_canonical_sensitivity import gradient_diagnostics
                                    with np.load(canonical_results/(model+'_corrected_probe.npz'),allow_pickle=False) as saved:params=dict(coef=saved['coef'][0],intercept=saved['intercept'][0])
                                    diag=gradient_diagnostics(X,y,params['coef'],float(params['intercept']),settings['C'])
                                    s.require(abs(diag['objective']-f['objective'])<1e-14 and abs(diag['gradient_infinity_norm']-f['gradient_infinity_norm'])<1e-12,'canonical numeric fitting identity mismatch')
                                    diagnostics=dict(f,reused_canonical=True,canonical_probe=s.identity((canonical_results/(model+'_corrected_probe.npz')).read_bytes()))
                                else:params,diagnostics=methods.fit(method,X,y,rows,settings['C'],layer)
                                finite=all(np.isfinite(v).all() for v in params.values() if np.asarray(v).dtype.kind not in 'US')
                                details.update(diagnostics,finite_parameters=bool(finite))
                                details['valid_for_scoring']=bool(details['valid_for_scoring'] and finite)
                                details['status']='converged' if details['valid_for_scoring'] else 'flagged'
                            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                                details.update(status='failed',valid_for_scoring=False,error=repr(exc),traceback=traceback.format_exc())
                            details['elapsed_seconds']=time.monotonic()-started
                            checkpoint(directory,identity,method,params,details)
                            report=json.loads((directory/'fit.json').read_text())
                        print(name+' '+report['status']+' '+str(round(report['elapsed_seconds'],2))+'s',flush=True)
                        report['id']=name;reports.append(report)
                        if report['valid_for_scoring']:
                            with np.load(directory/'parameters.npz',allow_pickle=False) as archive:params={k:archive[k] for k in archive.files}
                            path=output/'scores'/(name+'.npz')
                            if path.exists():
                                with np.load(path,allow_pickle=False) as archive:
                                    for endpoint in ('atomic','bare'):s.require(np.array_equal(archive[endpoint],methods.score(method,params,arrays[endpoint])),'checkpoint score mismatch')
                            else:
                                np.savez_compressed(path,atomic=methods.score(method,params,arrays['atomic']),bare=methods.score(method,params,arrays['bare']))
                        del X
                del arrays
        (output/'fits.json').write_bytes(s.json_bytes(reports))
        (output/'layer_bindings.json').write_bytes(s.json_bytes(layer_bindings))
        print('All 72 configurations checkpointed; evaluating paired withholding differences',flush=True)
        records,draws,missing=evaluate(bundle,config,output,reports)
    elapsed=time.monotonic()-start
    triggers=[r for r in records if r['triggered']]
    (output/'metrics.json').write_bytes(s.json_bytes(records))
    flat=[]
    for record in records:
        r={k:v for k,v in record.items() if not isinstance(v,dict)};r.update(record['paired_delta_ci']);flat.append(r)
    (output/'metrics.csv').write_bytes(s.csv_bytes(flat))
    np.savez_compressed(output/'scores'/'paired_draws.npz',full=np.array([a for a,b in draws]).T,P15=np.array([b for a,b in draws]).T)
    (output/'threshold_triggers.json').write_bytes(s.json_bytes(triggers))
    bundle.bundle.unchanged();staging.verify(root,prepared,exports)
    s.require(binding['source_sha256']=={name:s.sha((root/name).read_bytes()) for name in SOURCES},'code changed during pilot')
    receipt=dict(status='complete' if not missing and all(r['valid_for_scoring'] for r in reports) else 'completed_with_failures',
        completed_utc=datetime.now(timezone.utc).isoformat(),correction_commit=s.CORRECTION_COMMIT,local_platform=platform.system(),hostname=platform.node(),
        unique_configurations=len(reports),reused_canonical=sum(r['reused_canonical'] for r in reports),
        new_configuration_fits=sum(not r['reused_canonical'] for r in reports),failures=[r['id'] for r in reports if not r['valid_for_scoring']],
        endpoint_summaries=len(records),missing_comparisons=missing,threshold_triggers=len(triggers),elapsed_seconds=elapsed,
        fit_seconds=sum(r['elapsed_seconds'] for r in reports),cpu_concurrency=1,blas_threads=1,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__),
        bootstrap=config['bootstrap'],P_A_frozen=False,production_bank_fits=0,repair_fits=0,final_test_access=False,
        artifacts={str(p.relative_to(output)):s.identity(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'completion.json').write_bytes(s.json_bytes(receipt));return receipt


def evaluate(bundle,config,output,reports):
    atom=bundle.atomic;raw=bundle.raw
    atom_w=atomic_weights(atom,np.ones(len(atom),bool),config['bootstrap'])
    comp_w,schedule=compound_weights(raw,np.ones(len(raw),bool),config['bootstrap'])
    (output/'bootstrap_binding.json').write_bytes(s.json_bytes(dict(compound_schedule_sha256=schedule,atomic_weight_sha256=s.sha(atom_w.tobytes()))))
    reports={r['id']:r for r in reports};records=[];draws=[];missing=[]
    for model,settings in config['models'].items():
        for layer in settings['layers']:
            for method in config['methods']:
                ids=[f'{model}_L{layer}_{method}_{c}' for c in ('full','P15')]
                if not all(reports[k]['valid_for_scoring'] for k in ids):missing.append(dict(model=model,layer=layer,method=method,reason='flagged or failed fit'));continue
                with np.load(output/'scores'/(ids[0]+'.npz'),allow_pickle=False) as a,np.load(output/'scores'/(ids[1]+'.npz'),allow_pickle=False) as b:
                    for definition in config['endpoints']:
                        if definition['condition']=='atomic_D':
                            key='atomic';labels=np.array([int(r['label']) for r in atom]);mask=np.ones(len(atom),bool);topics=np.array([r['topic'] for r in atom]);weights=atom_w
                        else:
                            key='bare';labels=raw.cell.isin(definition['positive']).to_numpy(int)
                            mask=((raw.operator==definition['operator']) & raw.cell.isin(definition['positive']+definition['negative'])).to_numpy()
                            topics=raw.topic.to_numpy();weights=comp_w
                        found,delta=compare_endpoint(a[key],b[key],labels,mask,np.ones(len(labels),bool),topics,weights,definition,model,config['bootstrap'],config['topics'])
                        for r in found:
                            r.update(layer=layer,method=method,full_auroc=r.pop('historical_retained_auroc'),P15_auroc=r.pop('corrected_retained_auroc'),withholding_delta=r.pop('training_delta'))
                            for field in ('historical_full_auroc','coverage_delta','total_delta'):r.pop(field)
                            r['full_ci']=r.pop('historical_ci');r['P15_ci']=r.pop('corrected_ci')
                            threshold=config['review_drop_thresholds'].get(r['metric'])
                            r['review_drop_threshold']=threshold
                            r['triggered']=threshold is not None and r['withholding_delta'] is not None and r['withholding_delta'] < -threshold
                            if r['status']!='computed' or r['paired_delta_ci']['ci_status']!='ok':missing.append(dict(model=model,layer=layer,method=method,reason='insufficient endpoint uncertainty',metric=r['metric'],scope=r['scope'],topic=r['topic']))
                        records.extend(found);draws.extend(delta)
                print(f'Evaluated {model} L{layer} {method}',flush=True)
    return records,draws,missing
