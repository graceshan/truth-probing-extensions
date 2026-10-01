"""Historical development results from finalized scores, with no model execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import priority2_input_controls as p
from src import pinned_method_evaluation as methods_eval
from src import priority2_evaluation as controls_eval
from src import llama_replication_evaluation as llama_eval
from src.clean_transfer_evaluation import match_and_or, finite_records, parse_bool
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, AUC, interval
from src.t1_score_package import Package, load_inputs, read_json, EVAL_DIRS, SCORE_DIRS, SPECS

LABEL = 'DEVELOPMENT, pre-label/alias-cleanup; historical definitions; no correction overlay'
ATOL = 1e-12


def surface_cells(frame):
    """Canonical TF becomes surface FT under BA (and canonical FT becomes TF)."""
    c.require(frame.ordering.isin(['AB','BA']).all(), 'unknown surface ordering')
    a, b = parse_bool(frame.canonical_truth_a), parse_bool(frame.canonical_truth_b)
    first = np.where(frame.ordering == 'AB',a,b)
    second = np.where(frame.ordering == 'AB',b,a)
    for column, expected in [('surface_first_truth',first),('surface_second_truth',second)]:
        if column in frame:
            c.require(np.array_equal(parse_bool(frame[column]),expected), 'surface label mismatch')
    return np.char.add(np.where(first,'T','F'),np.where(second,'T','F'))


def compute_lr(frame, spec):
    schedule = EntityBootstrap(frame,spec['bootstrap'])
    plan = MetricPlan(frame,match_and_or(frame),spec)
    points = plan.evaluate(np.ones((1,len(frame))))[0]
    draws = np.empty((spec['bootstrap']['replicates'],len(plan.records)))
    for start in range(0,len(draws),64):
        stop = min(start+64,len(draws))
        draws[start:stop] = plan.evaluate(schedule.row_weights(start,stop))
    table = pd.DataFrame([{**record,'estimate':points[i],**interval(draws[:,i],spec['bootstrap'])}
                          for i,record in enumerate(plan.records)])
    return table,draws,schedule


def diagnostics(frame, options):
    """Surface-position and separately labeled canonical-cell AUROCs, shared draws."""
    frame = frame.sort_values('example_id').reset_index(drop=True)
    schedule = EntityBootstrap(frame,options)
    scores = frame.frozen_probe_score.to_numpy(float)
    definitions = [('mixed_first_true_vs_second_true',['TF'],['FT']),
                   ('first_true_vs_first_false',['TT','TF'],['FT','FF']),
                   ('second_true_vs_second_false',['TT','FT'],['TF','FF']),
                   ('tf_vs_ff',['TF'],['FF']),('ft_vs_ff',['FT'],['FF'])]
    records=[]
    for labeling,cells in [('surface',surface_cells(frame)),('canonical',frame.cell.to_numpy())]:
        for operator in ['AND','OR']:
            for ordering in ['all','AB','BA']:
                mask = (frame.operator == operator).to_numpy(copy=True)
                if ordering != 'all': mask &= (frame.ordering == ordering).to_numpy()
                for name,positive,negative in definitions:
                    pos = np.isin(cells,positive)
                    idx = np.flatnonzero(mask & (pos | np.isin(cells,negative)))
                    fn = AUC(idx,scores[idx],pos[idx])
                    samples=np.concatenate([fn(schedule.row_weights(start,min(start+64,options['replicates'])))
                                            for start in range(0,options['replicates'],64)])
                    metric_name=name if labeling == 'surface' else name.replace('mixed_first_true_vs_second_true','canonical_tf_vs_ft').replace('first_true_vs_first_false','canonical_a_true_vs_false').replace('second_true_vs_second_false','canonical_b_true_vs_false')
                    records.append(dict(labeling=labeling,operator=operator,ordering=ordering,metric=metric_name,
                        estimate=float(fn(np.ones((1,len(frame))))[0]),**interval(samples,options),
                        interpretation='descriptive constituent association, not formal compound accuracy'))
    return pd.DataFrame(records)


def condition_frames(data):
    """Only non-Boolean score conditions receive surface AUROC diagnostics."""
    result={('qwen_lr','raw_reference'):data['qwen']}
    for (group,method),part in data['methods'].groupby(['analysis_group','method']):
        result['methods',group+'/'+method] = data['frame'].merge(part[['example_id','frozen_probe_score']],on='example_id',validate='one_to_one')
    for family,base in [('controls',data['qwen']),('llama',data['llama'])]:
        result[family,p.RAW_ID] = base
        for condition in ([p.BOTH,p.LEAST] if family == 'controls' else [p.BOTH]):
            if family == 'controls':
                lookup = data['controls'].query('condition_id == @condition').set_index('base_example_id').frozen_probe_score
            else:
                values=data['llama_scores'].set_index('example_id').frozen_probe_score
                lookup=data['tables'][condition].set_index('base_example_id').example_id.map(values)
            copy=base.copy(); mask=copy.operator == 'OR'
            copy.loc[mask,'frozen_probe_score']=copy.loc[mask,'example_id'].map(lookup)
            result[family,condition]=copy
        lookup=(data['isolated'].set_index('fact_key').frozen_probe_score if family == 'controls' else
                data['llama_scores'].set_index('example_id').frozen_probe_score)
        mapped=base.merge(data['mapping'],left_on='example_id',right_on='base_example_id',validate='one_to_one')
        continuous,_=controls_eval.composition(mapped.fact_a_key.map(lookup),mapped.fact_b_key.map(lookup),mapped.operator)
        result[family,p.EXTERNAL]=base.assign(frozen_probe_score=continuous)
    return result


def compare_saved(package, family, table, draws, schedule):
    prefix=EVAL_DIRS[family]
    manifest=package.json(prefix+'/evaluation_manifest.json')
    c.require(schedule.sha256 == manifest['schedule_sha256'], 'recorded schedule hash mismatch: '+family)
    with np.load(package.artifact(prefix+'/bootstrap_pair_weights.npz'),allow_pickle=False) as saved:
        c.require(saved['pair_ids'].tolist() == schedule.pair_ids and np.array_equal(saved['weights'],schedule.weights),
                  'recorded pair weights mismatch: '+family)
    previous=package.csv(prefix+'/bootstrap_summary.csv')
    c.require(previous.metric_id.is_unique and table.metric_id.is_unique, 'duplicate result identity')
    differences=[]
    missing=sorted(set(previous.metric_id)-set(table.metric_id)); extra=sorted(set(table.metric_id)-set(previous.metric_id))
    for key in missing: differences.append(dict(family=family,metric_id=key,field='metric_id',reason='missing rebuilt metric'))
    for key in extra: differences.append(dict(family=family,metric_id=key,field='metric_id',reason='unexpected rebuilt metric'))
    old=previous.set_index('metric_id'); new=table.set_index('metric_id')
    common=sorted(set(old.index)&set(new.index)); maximum=0.; roundoff=0
    for field in ['estimate','ci_low','ci_high','valid_fraction','total_replicates','valid_replicates','invalid_replicates']:
        a=pd.to_numeric(new.loc[common,field].replace('',np.nan)).to_numpy(float)
        b=pd.to_numeric(old.loc[common,field].replace('',np.nan)).to_numpy(float)
        finite=np.isfinite(a)&np.isfinite(b)
        delta=np.abs(a[finite]-b[finite]); maximum=max(maximum,float(delta.max(initial=0)))
        roundoff+=int(np.count_nonzero((delta>0)&(delta<=ATOL)))
        for idx in np.flatnonzero(~np.isclose(a,b,atol=ATOL,rtol=0,equal_nan=True)):
            differences.append(dict(family=family,metric_id=common[idx],field=field,
                rebuilt=None if not np.isfinite(a[idx]) else float(a[idx]),saved=None if not np.isfinite(b[idx]) else float(b[idx])))
    for key in common:
        if old.loc[key,'ci_status'] != new.loc[key,'ci_status']:
            differences.append(dict(family=family,metric_id=key,field='ci_status',saved=old.loc[key,'ci_status'],rebuilt=new.loc[key,'ci_status']))
    with np.load(package.artifact(prefix+'/bootstrap_draws.npz'),allow_pickle=False) as saved:
        try:
            saved_ids=saved['metric_ids'].tolist()
            id_binding='non-object metric_ids array'
        except ValueError as exc:
            # The original LR writer stored pandas object strings. Never unpickle.
            if 'Object arrays cannot be loaded' not in str(exc): raise
            saved_ids=previous.metric_id.tolist()
            id_binding='hash-verified companion bootstrap_summary.csv order; object metric_ids not deserialized'
            c.require(table.metric_id.tolist() == saved_ids, 'companion summary order differs')
        c.require(set(saved_ids) == set(table.metric_id) and len(saved_ids) == len(table), 'saved draw identity coverage')
        ordered=draws[:,[table.metric_id.tolist().index(key) for key in saved_ids]]
        values=saved['values']; validity=np.isfinite(ordered)
        c.require(values.shape == ordered.shape, 'saved draw shape mismatch')
        mask_difference=int(np.count_nonzero(validity != saved['valid']))
        draw_differences=int(np.count_nonzero(~np.isclose(ordered,values,atol=ATOL,rtol=0,equal_nan=True)))
        both=np.isfinite(ordered)&np.isfinite(values)
        max_draw_delta=float(np.abs(ordered[both]-values[both]).max(initial=0))
        if mask_difference or draw_differences:
            differences.append(dict(family=family,field='bootstrap_draws',validity_differences=mask_difference,
                                    differing_values=draw_differences,max_absolute_difference=max_draw_delta))
    summary=dict(family=family,metrics=len(table),saved_metrics=len(previous),differences=len(differences),
        absolute_tolerance=ATOL,max_table_absolute_difference=maximum,roundoff_cells_within_tolerance=roundoff,
        max_draw_absolute_difference=max_draw_delta,draw_values_compared=int(draws.size),
        saved_draw_id_binding=id_binding,
        schedule_sha256=schedule.sha256,schedule_and_pair_weights_verified=True,
        minimum_effective_replicates=int(table.valid_replicates.min()),maximum_effective_replicates=int(table.valid_replicates.max()))
    return summary,differences


def atomic_aggregates(package, specs):
    """Read recorded aggregates, cross-check structured tables, never invent row scores."""
    result=[]
    for family,prefix in [('qwen_lr','atomic_probe_selection/qwen25_7b'),('llama','llama31_replication_v1/probe')]:
        selection=package.json(prefix+'/selection.json'); saved=package.csv(prefix+'/validation_metrics.csv')
        selected=saved[(saved.layer == selection['selected_layer']) & (saved.C == selection['selected_C'])]
        c.require(len(selected) == 1, 'atomic selection aggregate coverage')
        expected=specs[family].get('expected_selection',specs[family].get('selected'))
        c.require(selection['selected_layer'] == expected['layer'] and selection['selected_C'] == expected['C'], 'atomic selection binding')
        for metric,value in selection['selected_validation_metrics'].items():
            if not metric.startswith('validation_'): continue
            c.require(np.isclose(selected.iloc[0][metric],value,rtol=0,atol=ATOL), 'atomic saved aggregates disagree')
            result.append(dict(family=family,condition='selected_l2_logistic',layer=selection['selected_layer'],
                metric=metric,estimate=value,source=prefix+'/selection.json',evidence='saved_aggregate_only',
                row_score_recomputed=False,ci_status='unavailable_no_atomic_row_scores'))
    for suite,suffix in [('faithful','atomic_method_suite_pinned_v1'),('matched','atomic_method_matched_pinned_v1')]:
        prefix='pinned_atomic_methods_qwen25_v1/'+suffix+'/qwen25_7b'
        summary=package.json(prefix+'/summary.json'); table=package.csv(prefix+'/validation_metrics.csv')
        c.require(summary['atomic_test_accessed'] is False and summary['compound_data_accessed'] is False, 'atomic suite scope')
        for comparison,methods in summary['comparisons'].items():
            for method,record in methods.items():
                row=table[(table.method == method)&(table.layer == record['layer'])]
                c.require(len(row) == 1,'atomic method aggregate coverage')
                for metric,value in record.items():
                    if not metric.startswith('validation_') or 'sha256' in metric or metric == 'validation_rows': continue
                    c.require(np.isclose(row.iloc[0][metric],value,atol=ATOL,rtol=0), 'atomic suite aggregates disagree')
                    result.append(dict(family='methods',condition=suite+'/'+comparison+'/'+method,layer=record['layer'],
                        metric=metric,estimate=value,source=prefix+'/summary.json',evidence='saved_aggregate_only',
                        row_score_recomputed=False,ci_status='unavailable_no_atomic_row_scores'))
    return pd.DataFrame(result)


def provenance(package, data):
    differences=[]; checked=[]
    def walk(value,location):
        if isinstance(value,dict):
            for key,child in value.items():
                if key.endswith('.py') and isinstance(child,str) and len(child) == 64:
                    path=package.repo/key
                    current=c.file_hash(path) if path.is_file() else None
                    record=dict(source=location,path=key,recorded_sha256=child,checkout_sha256=current)
                    checked.append(record)
                    if current != child: differences.append(record)
                else: walk(child,location+'/'+key)
        elif isinstance(value,list):
            for i,child in enumerate(value): walk(child,location+'/'+str(i))
    for family in SPECS:
        walk(data['manifests'][family],family+'/scoring_manifest')
        walk(data['evaluations'][family],family+'/evaluation_manifest')
        walk(data['specs'][family],family+'/analysis_spec')
    binding=package.json(SCORE_DIRS['qwen_lr']+'/representation_binding.json')
    pinned=binding['producer_source_audit']['files']['src/pinned_atomic_probes.py']
    c.require(pinned['current_sha256'] != pinned['recorded_sha256'], 'expected historical conflict missing')
    return dict(source_hash_comparisons=checked,checkout_differences=differences,
        pinned_atomic_probes_historical_conflict=dict(**pinned,
            checkout_sha256=c.file_hash(package.repo/'src/pinned_atomic_probes.py'),status='unresolved',
            interpretation='Recorded working-file hash differs from recorded producer/Git hash. Matching restored score bytes does not resolve which code executed.'),
        upstream_verification='not repeated: activation extraction, probe fitting, atomic row-score evaluation, model/tokenizer execution',
        raw_source_paths='historical identity strings only; never accessed')


def write_json(path,value):
    with path.open('x') as handle: json.dump(value,handle,indent=2,allow_nan=False); handle.write('\n')


def readable_table(frame, columns):
    def text(value):
        if isinstance(value,(float,np.floating)): return '' if not np.isfinite(value) else f'{value:.9f}'
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
        ['| '+' | '.join(text(row[col]) for col in columns)+' |' for _,row in frame.iterrows()])


def run(payload_root, repo_root, output_root):
    repo=Path(repo_root).resolve(); payload=Path(payload_root).resolve(); output=Path(output_root).resolve()
    c.require(not output.exists() and not output.is_relative_to(payload.parent) and not output.is_relative_to(repo),
              'output must be new and outside the checkout and backup directory')
    output.mkdir(parents=True,exist_ok=False)
    try:
        print('Verifying package, frozen specs, benchmark, and score joins...',flush=True)
        package=Package(payload,repo); data=load_inputs(package); specs=data['specs']
        options=specs['qwen_lr']['bootstrap']
        c.require(options['replicates'] == 2000 and options['seed'] == 1729 and options['rng'] == 'numpy.random.Generator(PCG64)' and
                  options['pair_weight'] == 'm_i*m_j' and options['minimum_valid'] == 1800 and
                  options['undefined'] == 'retain_invalid_no_redraw_no_imputation', 'historical bootstrap policy required')
        atomic=atomic_aggregates(package,specs); source_audit=provenance(package,data)
        tables=[]; comparisons=[]; discrepancies=[]
        with threadpool_limits(limits=1):
            for family in SPECS:
                print('Rebuilding '+family+' (2000 shared endpoint replicates)...',flush=True)
                if family == 'qwen_lr': table,draws,schedule=compute_lr(data['qwen'],specs[family])
                elif family == 'methods': table,draws,schedule=methods_eval.compute(data['methods'],data['frame'],specs[family])
                elif family == 'controls': table,draws,schedule,_=controls_eval.compute(data['qwen'],data['tables'],data['mapping'],data['controls'],data['isolated'],specs[family])
                else: table,draws,schedule=llama_eval.compute(data['llama'],data['tables'],data['mapping'],data['llama_scores'],specs[family]['policy'])
                summary,diff=compare_saved(package,family,table,draws,schedule)
                comparisons.append(summary); discrepancies.extend(diff)
                np.savez_compressed(output/(family+'_bootstrap_draws.npz'),metric_ids=table.metric_id.to_numpy(str),values=draws,valid=np.isfinite(draws))
                table['family']=family; table['result_stage']=LABEL; table['evidence']='recomputed_from_row_scores'
                table['truth_cell_basis']='canonical'
                tables.append(table)
                del draws
            print('Computing surface-position diagnostics, separately from canonical truth cells...',flush=True)
            surfaces=[]
            for (family,condition),frame in condition_frames(data).items():
                surfaces.append(diagnostics(frame,options).assign(family=family,condition=condition,result_stage=LABEL))
        metrics=pd.concat(tables,ignore_index=True); surface=pd.concat(surfaces,ignore_index=True)
        atomic['result_stage']=LABEL
        for name,table in [('metrics',metrics),('surface_diagnostics',surface),('atomic_validation',atomic)]:
            table.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
            write_json(output/(name+'.json'),finite_records(table))
        # Shared endpoint multiplicities are included for direct auditing, not just saved hashes.
        np.savez_compressed(output/'bootstrap_schedule.npz',pair_ids=np.asarray(schedule.pair_ids),weights=schedule.weights,
            **{topic+'_multiplicities':v for topic,v in schedule.multiplicities.items()})
        write_json(output/'input_hashes.json',package.inputs)
        write_json(output/'discrepancies.json',dict(absolute_tolerance=ATOL,comparisons=comparisons,metric_discrepancies=discrepancies,
            surface_note='New surface diagnostics use constituent labels after ordering; original canonical metrics are retained and not overwritten.',
            provenance=source_audit))
        package.unchanged()
        atomic_gap='Atomic validation row scores are not packaged. Values are hash-verified structured saved aggregates cross-checked against validation_metrics.csv; no atomic CI is reconstructed.'
        summary=dict(result_stage=LABEL,status='complete_with_evidence_gaps',score_package_verified=True,
            repo_root=str(repo),payload_root=str(payload),output_root=str(output),
            recovery_commit='21996d9ba5b6b217a07a06d33c3d56db908c3b41',
            checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            completed_utc=datetime.now(timezone.utc).isoformat(),runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
            inventory_files_verified=len(package.inventory),benchmark_rows=len(data['frame']),
            benchmark_pairs=data['frame'].pair_id.nunique(),metrics=len(metrics),surface_diagnostics=len(surface),atomic_aggregates=len(atomic),
            bootstrap=options,comparisons=comparisons,metric_discrepancies=len(discrepancies),
            source_hash_differences=len(source_audit['checkout_differences']),historical_pinned_conflict='unresolved',
            evidence_gaps=[atomic_gap,'Activation tensors excluded; upstream extraction/fitting checks cannot be repeated.',
                'Recovered code/registry preservation by mac 1 is separate from this score package.',
                'Historical source hash conflicts remain unresolved; byte-level score verification is not proof of upstream execution provenance.'],
            upstream_missing_records=package.gaps,checks=package.checks,
            invariants=dict(backup_unchanged=True,model_fitting=False,activation_arrays_opened=False,final_test_inspected=False,correction_overlay_applied=False))
        code={}
        for name in ['src/t1_score_package.py','src/t1_score_rebuild.py','scripts/46_rebuild_t1_development_results.py',
                     'src/clean_transfer_statistics.py','src/clean_transfer_evaluation.py','src/pinned_method_evaluation.py',
                     'src/priority2_evaluation.py','src/llama_replication_evaluation.py','src/priority2_input_controls.py']:
            code[name]=c.record(repo/name)
        summary['rebuild_code']=code
        summary['outputs']={path.name:c.record(path) for path in sorted(output.iterdir())}
        write_json(output/'verification.json',summary)
        primary=metrics[(metrics.scope == 'pooled') & metrics.metric.isin(['and_auroc','or_auroc','or_mixed_vs_ff_auroc'])].copy()
        primary['condition']=primary.apply(lambda row: row.get('condition_id') if pd.notna(row.get('condition_id')) else
            '/'.join(str(row[x]) for x in ['analysis_group','method']) if pd.notna(row.get('method')) else 'raw_reference',axis=1)
        # Only concise tables go into the Git summary; full metrics and draws stay external.
        main=primary[primary.category != 'paired_contrast']
        surface_main=surface[(surface.family.isin(['qwen_lr','llama'])) & (surface.condition == p.RAW_ID) &
            (surface.operator == 'OR') & surface.metric.isin(['mixed_first_true_vs_second_true','canonical_tf_vs_ft','tf_vs_ff','ft_vs_ff'])]
        lines=['# T1 historical development-results rebuild','',LABEL,'',
            f'Verified {len(package.inventory)} payload files; {len(data["frame"])} benchmark rows / 524 pairs / 262 entities.',
            f'Rebuilt {len(metrics)} metric rows. Discrepancies beyond absolute tolerance {ATOL}: {len(discrepancies)}.',
            'All four recorded schedule hashes and saved pair-weight arrays matched. Effective replicate counts are reported per result.',
            '', '## Main score results','',readable_table(main,['family','condition','metric','estimate','ci_low','ci_high','valid_replicates']),
            '', '## Atomic validation (saved aggregates only)','',
            readable_table(atomic[atomic.metric == 'validation_overall_auroc'],['family','condition','layer','estimate','evidence']),
            '', '## OR constituent diagnostics (surface versus canonical)','',
            readable_table(surface_main,['family','labeling','ordering','metric','estimate','ci_low','ci_high','valid_replicates']),
            '', '## Evidence limits','',atomic_gap,'',
            'The recorded pinned_atomic_probes.py working-file digest conflicts with the recorded producer/Git digest. It remains unresolved. See discrepancies.json.',
            'Activation tensors are excluded. Upstream extraction and fitting are not revalidated. No final-test rows or correction overlay were used.',
            '', 'Surface-position results are in surface_diagnostics.csv/json. Canonical TF becomes surface FT under BA; canonical-cell results remain separately labeled.',
            '', 'Full input identities, available and missing upstream byte records, provenance differences, and checks are in verification.json, input_hashes.json, and discrepancies.json.', '']
        (output/'summary.md').write_text('\n'.join(lines))
        print(json.dumps({key:summary[key] for key in ['status','output_root','metrics','metric_discrepancies','source_hash_differences']},indent=2),flush=True)
        return summary
    except Exception as exc:
        write_json(output/'failure.json',dict(status='failed',error=type(exc).__name__,message=str(exc),result_stage=LABEL))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-root',type=Path,required=True,help='Read-only restored staging payload containing inventory.json')
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1],help='Recovered checkout with frozen configs and benchmark')
    parser.add_argument('--output-root',type=Path,required=True,help='New directory outside checkout and backup')
    args=parser.parse_args()
    run(args.payload_root,args.repo_root,args.output_root)
