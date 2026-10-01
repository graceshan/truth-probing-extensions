"""Only the two reviewed v3 full-outer-training canonical references; no reservation."""
import io
import json
import platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from src import selection_repair_sensitivity_v3 as s
from src import clean_atomic_probes as canonical_lr
from src.selection_repair_canonical_sensitivity import (fit_reference, atomic_weights, compound_weights,
                                                       compare_endpoint, refresh_triggers)
from src.selection_repair_cache_export_remote import DIRECTORIES
from src.selection_repair_sensitivity_verification import verify_exports
from src.clean_transfer_statistics import AUC


class BoundInputs:
    def __init__(self,root,payload,prepared,artifacts):
        self.root,self.payload,self.prepared,self.artifacts=map(Path,(root,payload,prepared,artifacts))
        self.config,self.inventory=s.verify_preparation(root,prepared)
        s.require(self.artifacts.resolve()==Path(self.inventory['physical_export_root']), 'wrong physical export directory')
        verify_exports(root,self.inventory['parent_prepared_root'],artifacts)
        self.request=json.loads((self.prepared/'export_request.json').read_text())['caches']
        self.remote=json.loads((self.artifacts/'export_receipt.json').read_text())['caches']
        self.decision=json.loads((self.root/self.config['adoption_path']).read_text())
        physical=json.loads((self.root/self.decision['bindings']['physical_receipt']['path']).read_text())
        inventory_pin=next(r for r in physical['input_identities'] if r['location']=='inventory.json')
        self.package=s.Inputs(root,payload,inventory_pin)
        self.arrays={};self.metadata_used={}

    def read(self,relative):
        record=self.inventory['metadata_inputs'][s.ROOT_REMOTE+relative]
        data=(self.payload/record['archive_path']).read_bytes()
        s.require(s.identity(data)=={k:record[k] for k in ('bytes','sha256')}, 'bound input changed: '+relative)
        self.metadata_used[relative]=record
        return data

    def array(self,key):
        if key in self.arrays:return self.arrays[key]
        if self.remote[key]['status']!='verified':return None
        path=self.artifacts/(key+'.npy')
        s.require(s.identity(path.read_bytes())=={k:self.remote[key]['export'][k] for k in ('bytes','sha256')}, 'selected export changed')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        s.require(array.shape==tuple(self.remote[key]['export']['shape']) and array.dtype==np.float16
                  and array.flags.c_contiguous and np.isfinite(array).all(), 'invalid selected states')
        self.arrays[key]=array
        return array

    def score(self,model,w,b):
        result={};lookup={}
        with threadpool_limits(limits=1):
            val=self.array(model+'_validation')
            if val is not None:result['atomic_D']=np.asarray(val,dtype=np.float64)@w+b
            keys=['qwen_raw','qwen_controls'] if model=='qwen' else ['llama_transfer']
            for key in keys:
                values=self.array(key)
                if values is None:continue
                rows=s.parse_rows(self.read(DIRECTORIES[key]+'/metadata.csv'))
                ids=[rows[i]['example_id'] for i in self.request[key]['row_indices']]
                s.require(len(ids)==len(values) and len(set(ids))==len(ids) and not set(ids)&set(lookup), 'score identity collision')
                lookup.update(zip(ids,np.asarray(values,dtype=np.float64)@w+b))
        raw=s.parse_rows(self.read(s.RAW+'metadata.csv'));raw_ids=[r['example_id'] for r in raw]
        both=s.parse_rows(self.read(s.GEN+'or_explicit_or_both_v1.csv'))
        mapping={r['base_example_id']:r for r in s.parse_rows(self.read(s.GEN+'constituent_map.csv'))}
        facts=s.parse_rows(self.read(s.GEN+'isolated_facts.csv'))
        if all(e in lookup for e in raw_ids):result['raw_reference']=np.array([lookup[e] for e in raw_ids])
        both_by_base={r['base_example_id']:r['example_id'] for r in both}
        if all(e in lookup for e in both_by_base.values()):
            result['or_explicit_or_both_v1']=np.array([lookup[both_by_base[e]] if e in both_by_base else 0. for e in raw_ids])
        if all(f['fact_key'] in lookup for f in facts):
            first=np.array([lookup[mapping[e]['fact_a_key']] for e in raw_ids])
            second=np.array([lookup[mapping[e]['fact_b_key']] for e in raw_ids])
            result['isolated_external_minmax_v1']=np.where([r['operator']=='AND' for r in raw],np.minimum(first,second),np.maximum(first,second))
        return result,lookup

    def historical(self,model,atomic):
        settings=self.config['models'][model]
        prefix='atomic_probe_selection/qwen25_7b/' if model=='qwen' else 'llama31_replication_v1/probe/'
        selection=json.loads(self.read(prefix+'selection.json'));data=self.read(prefix+'selected_probe.npz')
        s.require(selection['selected_layer']==settings['layer'] and selection['selected_C']==settings['C']
                  and selection['preprocessing']=='none' and selection['probe_configuration']==canonical_lr.PROBE_CONFIG
                  and selection['optimization_policy']==canonical_lr.OPTIMIZATION_POLICY
                  and selection['selected_probe_sha256']==s.sha(data) and selection['all_final_fits_converged'] is True, 'historical probe contract')
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            w=archive['coef'][0].copy();b=float(archive['intercept'][0])
            s.require(archive['coef'].shape==(1,settings['width']) and archive['intercept'].shape==(1,)
                      and np.array_equal(archive['classes'],[0,1]) and int(archive['layer'])==settings['layer']
                      and float(archive['C'])==settings['C'] and np.isfinite(w).all() and np.isfinite(b), 'historical probe archive')
        scores,lookup=self.score(model,w,b);checks=[]
        if 'atomic_D' in scores:
            labels=np.array([int(r['label']) for r in atomic]);old=scores['atomic_D']
            full=float(AUC(np.arange(len(labels)),old,labels)(np.ones((1,len(labels))))[0])
            recorded=selection['selected_validation_metrics']['validation_overall_auroc']
            s.require(abs(full-recorded)<1e-12, 'historical atomic AUROC replay mismatch')
            checks.append(dict(endpoint='atomic_D',rows=len(labels),full_auroc=full,recorded_auroc=recorded,verified=True))
        groups=([('pinned_transfer_qwen25_v1/scoring/','qwen_spec','row_scores.csv','row_scores'),
                 ('priority2_input_controls_v1/scoring/','qwen_controls_spec','condition_scores.csv','outputs'),
                 ('priority2_input_controls_v1/scoring/','qwen_controls_spec','isolated_scores.csv','outputs')]
                if model=='qwen' else [('llama31_replication_v1/analysis/scores/','llama_spec','scores.csv','score_file')])
        matched_ids=set()
        for prefix,spec_key,name,key in groups:
            manifest=json.loads(self.package.remote(prefix+'scoring_manifest.json'))
            s.require(manifest['complete'] is True and manifest['analysis_spec_sha256']==self.decision['bindings'][spec_key]['sha256'], 'historical score/spec mismatch')
            if 'selected_probe_sha256' in manifest:s.require(manifest['selected_probe_sha256']==s.sha(data), 'historical score/probe mismatch')
            expected=manifest['outputs'][name] if key=='outputs' else manifest[key]
            frame=pd.read_csv(io.BytesIO(self.package.remote(prefix+name,expected)),float_precision='round_trip')
            column='fact_key' if name=='isolated_scores.csv' else 'example_id'
            s.require(frame[column].is_unique, 'duplicate historical score identity')
            matched=[(eid,value) for eid,value in zip(frame[column],frame.frozen_probe_score) if eid in lookup]
            if matched:
                error=max(abs(lookup[eid]-float(value)) for eid,value in matched)
                s.require(error<1e-10, 'historical row-score replay mismatch')
                checks.append(dict(file=prefix+name,matched_rows=len(matched),maximum_absolute_error=error,verified=True))
                matched_ids.update(eid for eid,_ in matched)
        s.require(matched_ids==set(lookup), 'historical saved score coverage missing')
        return scores,dict(probe_sha256=s.sha(data),checks=checks)

    def unchanged(self):
        s.verify_preparation(self.root,self.prepared);self.package.unchanged()
        for name in self.metadata_used:self.read(name)


def run(root,payload,prepared,artifacts,output,score_output):
    root,payload,prepared,artifacts,output,score_output=map(Path,(root,payload,prepared,artifacts,output,score_output))
    s.require(platform.system()=='Darwin', 'fit/evaluation must remain on macOS')
    s.require(not output.exists() and not score_output.exists(), 'new v3 output directories required; no overwrite')
    s.require(not score_output.resolve().is_relative_to(root.resolve()), 'large score artifacts must remain outside Git')
    bundle=BoundInputs(root,payload,prepared,artifacts);config=bundle.config
    atomic=s.parse_rows((prepared/'atomic_D_eligibility.csv').read_bytes())
    pairs=s.parse_rows((prepared/'development_pair_eligibility.csv').read_bytes())
    raw=pd.read_csv(io.BytesIO(bundle.read(s.RAW+'metadata.csv')))
    raw['cell']=np.where(raw.canonical_truth_a,'T','F')+np.where(raw.canonical_truth_b,'T','F')
    atom_keep=np.array([r['current_eligible']=='True' for r in atomic])
    comp_keep=raw.pair_id.isin({r['pair_id'] for r in pairs if r['current_eligible']=='True'}).to_numpy()
    atom_w=atomic_weights(atomic,atom_keep,config['bootstrap']);comp_w,comp_schedule=compound_weights(raw,comp_keep,config['bootstrap'])
    # Historical replay of both models must pass before either corrected fit starts.
    historical={};baseline={}
    for model in config['models']:
        print('Verifying historical row-level replay: '+model,flush=True)
        historical[model],baseline[model]=bundle.historical(model,atomic)
    bundle.unchanged()
    output.mkdir(parents=True);score_output.mkdir(parents=True)
    (output/'historical_replay.json').write_bytes(s.json_bytes(baseline))
    fits={};missing=[];records=[];draws=[]
    for model,settings in config['models'].items():
        train=bundle.array(model+'_train')
        if train is None:
            missing.append(dict(model=model,endpoint='all',dependency=bundle.remote[model+'_train']));continue
        rows=s.parse_rows((prepared/(model+'_train.mapping_v3.csv')).read_bytes())
        indices=np.array([int(r['export_row_index']) for r in rows]);original=bundle.request[model+'_train']['row_indices']
        s.require([original[i] for i in indices]==[int(r['tensor_row_index']) for r in rows], 'fit original/export row mismatch')
        s.require(not ({r['source_row_id'] for r in rows}&s.QUARANTINED)
                  and all(r[s.STATUS]=='admitted' and r['split']=='train' for r in rows), 'nonadmitted fit row')
        y=np.array([int(r['label']) for r in rows]);X=np.asarray(train[indices],dtype=np.float64)
        print('Fitting '+model+' v3 canonical reference on '+str(len(y))+' rows',flush=True)
        probe,diagnostics=fit_reference(X,y,settings['C'],settings['layer']);del X
        fits[model]=dict(**diagnostics,training_rows=len(y),class_counts={str(k):int((y==k).sum()) for k in (0,1)},
            training_membership=s.identity((prepared/(model+'_train.mapping_v3.csv')).read_bytes()),
            source_tensor={k:bundle.request[model+'_train'][k] for k in ('path','sha256','bytes','shape')},
            source_export=bundle.remote[model+'_train']['export'],original_tensor_indices_sha256=s.sha(s.canonical([int(r['tensor_row_index']) for r in rows])),
            selected_export_indices_sha256=s.sha(s.canonical(indices.tolist())),C=settings['C'],saved_layer_index=settings['layer'],
            preprocessing='none',probe_configuration=canonical_lr.PROBE_CONFIG,optimization_policy=canonical_lr.OPTIMIZATION_POLICY,
            historical_probe_sha256=baseline[model]['probe_sha256'])
        # Save each attempt immediately; a later endpoint failure must not lose fit provenance.
        (output/(model+'_fit.json')).write_bytes(s.json_bytes(fits[model]))
        print(model+' iterations='+str(diagnostics['final_n_iter'])+' gradient_inf='+str(diagnostics['gradient_infinity_norm']),flush=True)
        np.savez_compressed(output/(model+'_corrected_probe.npz'),coef=probe.coef_,intercept=probe.intercept_,classes=probe.classes_,
                            layer=settings['layer'],C=settings['C'],n_iter=probe.n_iter_,max_iter=probe.max_iter)
        if not diagnostics['valid_for_scoring']:
            missing.append(dict(model=model,endpoint='all',dependency='canonical fit flagged; see diagnostics'));continue
        corrected,_=bundle.score(model,probe.coef_[0],float(probe.intercept_[0]))
        np.savez_compressed(score_output/(model+'_scores.npz'),**{'historical_'+k:v for k,v in historical[model].items()},
                            **{'corrected_'+k:v for k,v in corrected.items()})
        for definition in config['endpoints']:
            condition=definition['condition']
            if condition not in historical[model] or condition not in corrected:
                missing.append(dict(model=model,endpoint=condition+'/'+definition['metric'],dependency={k:v for k,v in bundle.remote.items()
                    if k.startswith(model) and v['status']!='verified'}));continue
            if condition=='atomic_D':
                labels=np.array([int(r['label']) for r in atomic]);mask=np.ones(len(atomic),bool)
                eligible=atom_keep;topics=np.array([r['topic'] for r in atomic]);weights=atom_w
            else:
                labels=raw.cell.isin(definition['positive']).to_numpy(int)
                mask=((raw.operator==definition['operator']) & raw.cell.isin(definition['positive']+definition['negative'])).to_numpy()
                eligible=comp_keep;topics=raw.topic.to_numpy();weights=comp_w
            found,delta=compare_endpoint(historical[model][condition],corrected[condition],labels,mask,eligible,topics,weights,
                                        definition,model,config['bootstrap'],config['topics'])
            for record in found:
                if record['status']!='computed' or record['paired_delta_ci']['ci_status']!='ok':
                    missing.append(dict(model=model,endpoint=condition+'/'+definition['metric'],scope=record['scope'],topic=record['topic'],
                                        dependency=record.get('dependency','Insufficient valid paired bootstrap replicates')))
            records.extend(found);draws.extend(delta)
    triggers,comparisons=refresh_triggers(records,draws,config)
    if draws:np.savez_compressed(score_output/'bootstrap_draws.npz',historical=np.array([d[0] for d in draws]).T,corrected=np.array([d[1] for d in draws]).T)
    compact=[]
    for record in records:
        item={k:v for k,v in record.items() if k not in ('historical_ci','corrected_ci','paired_delta_ci')}
        item.update(record['paired_delta_ci']);compact.append(item)
    if compact:(output/'metrics.csv').write_bytes(s.csv_bytes(compact))
    for name,value in [('metrics.json',records),('fits.json',fits),('rankings_and_contrasts.json',comparisons)]:
        (output/name).write_bytes(s.json_bytes(value))
    refresh=dict(triggered=bool(triggers),triggers=triggers,automatic_refresh_launched=False,
        affected_downstream=['Canonical raw LR compound transfer tables/figures and critical boundaries',
        'Priority-2 raw versus or-both and isolated min/max comparisons',
        'Qwen/Llama canonical-reference comparisons and conclusions',
        'Method comparisons reusing the canonical LR baseline or historical development eligibility'] if triggers else [])
    (output/'refresh_decisions.json').write_bytes(s.json_bytes(refresh));bundle.unchanged()
    receipt=dict(status='complete_sensitivity' if not missing and len(fits)==2 else 'incomplete_sensitivity',
        completed_utc=datetime.now(timezone.utc).isoformat(),local_platform=platform.system(),correction_commit=s.CORRECTION_COMMIT,
        adoption_sha256=config['adoption_sha256'],preparation_receipt=s.identity((prepared/'preparation_receipt.json').read_bytes()),
        export_receipt=s.identity((artifacts/'export_receipt.json').read_bytes()),endpoint_results=len(records),missing_endpoints=missing,
        historical_inputs=bundle.package.used,metadata_inputs=bundle.metadata_used,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__,blas_threads=1),
        bootstrap=dict(**config['bootstrap'],compound_schedule_sha256=comp_schedule,atomic_weight_sha256=s.sha(atom_w.tobytes())),
        fitting_operations=len(fits),production_bank_fits=0,reservations=0,new_model_extractions=0,final_test_predictions=0,
        artifacts={p.name:s.identity(p.read_bytes()) for p in sorted(output.iterdir())},
        external_artifacts={str(p):s.identity(p.read_bytes()) for p in sorted(score_output.iterdir())},
        source_sha256={name:s.sha((root/name).read_bytes()) for name in ['src/selection_repair_canonical_sensitivity_v3.py',
            'src/selection_repair_sensitivity_v3.py','src/selection_repair_canonical_sensitivity.py','src/clean_atomic_probes.py','src/clean_transfer_statistics.py']})
    (output/'sensitivity_receipt.json').write_bytes(s.json_bytes(receipt))
    return receipt
