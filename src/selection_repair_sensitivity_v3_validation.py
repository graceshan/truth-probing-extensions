"""Independent post-run AUROC/receipt checks; no activations, extraction or fitting."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import selection_repair_sensitivity_v3 as s


def check_identities(directory,expected):
    directory=Path(directory)
    for name,pin in expected.items():
        path=directory/name
        s.require(path.is_file() and s.identity(path.read_bytes())==pin, 'result artifact changed: '+name)


def check_values(actual,expected):
    s.require(np.allclose(actual,expected,rtol=0,atol=1e-12), 'independent metric/interval mismatch')


def validate(root,payload,prepared,results):
    root,payload,prepared,results=map(Path,(root,payload,prepared,results))
    config,inventory=s.verify_preparation(root,prepared)
    receipt=json.loads((results/'sensitivity_receipt.json').read_text())
    s.require(receipt['correction_commit']==s.CORRECTION_COMMIT and receipt['adoption_sha256']==config['adoption_sha256']
              and receipt['preparation_receipt']==s.identity((prepared/'preparation_receipt.json').read_bytes()), 'mixed result/preparation version')
    check_identities(results,receipt['artifacts'])
    for name,pin in receipt['external_artifacts'].items():check_identities(Path(name).parent,{Path(name).name:pin})
    for name,digest in receipt['source_sha256'].items():s.require(s.sha((root/name).read_bytes())==digest, 'result code changed')
    s.require(receipt['status']=='complete_sensitivity' and not receipt['missing_endpoints'] and receipt['fitting_operations']==2, 'sensitivity incomplete')
    metrics=json.loads((results/'metrics.json').read_text());fits=json.loads((results/'fits.json').read_text())
    def read(relative):
        record=inventory['metadata_inputs'][s.ROOT_REMOTE+relative]
        data=(payload/record['archive_path']).read_bytes()
        s.require(s.identity(data)=={k:record[k] for k in ('bytes','sha256')}, 'metric metadata changed')
        return data
    raw=pd.read_csv(io.BytesIO(read(s.RAW+'metadata.csv')))
    raw['cell']=np.where(raw.canonical_truth_a,'T','F')+np.where(raw.canonical_truth_b,'T','F')
    atomic=s.parse_rows((prepared/'atomic_D_eligibility.csv').read_bytes())
    pairs=s.parse_rows((prepared/'development_pair_eligibility.csv').read_bytes())
    atomic_keep=np.array([r['current_eligible']=='True' for r in atomic])
    compound_keep=raw.pair_id.isin({r['pair_id'] for r in pairs if r['current_eligible']=='True'}).to_numpy()
    score_files={Path(p).stem.removesuffix('_scores'):p for p in receipt['external_artifacts'] if p.endswith('_scores.npz')}
    bootstrap=next(p for p in receipt['external_artifacts'] if p.endswith('/bootstrap_draws.npz'))
    endpoints={ (r['condition'],r['metric']):r for r in config['endpoints'] }
    s.require(len(metrics)==receipt['endpoint_results']==len(endpoints)*len(config['models'])*(2+len(config['topics'])), 'endpoint coverage')
    seen=set();max_error=0.
    with np.load(bootstrap,allow_pickle=False) as archive:
        old_draws=archive['historical'];new_draws=archive['corrected']
        s.require(old_draws.shape==new_draws.shape==(config['bootstrap']['replicates'],len(metrics)), 'bootstrap shape')
    for model,settings in config['models'].items():
        rows=s.parse_rows((prepared/(model+'_train.mapping_v3.csv')).read_bytes())
        s.require(len(rows)==fits[model]['training_rows']==3044 and all(r[s.STATUS]=='admitted' for r in rows)
                  and not {r['source_row_id'] for r in rows}&s.QUARANTINED, 'actual fit membership')
        s.require(fits[model]['training_membership']==s.identity((prepared/(model+'_train.mapping_v3.csv')).read_bytes())
                  and fits[model]['valid_for_scoring'] and fits[model]['finite'] and fits[model]['gradient_infinity_norm']<=1e-4, 'invalid fit diagnostics')
        s.require(fits[model]['C']==settings['C'] and fits[model]['saved_layer_index']==settings['layer'], 'fit configuration mismatch')
        with np.load(score_files[model],allow_pickle=False) as archive:
            scores={k:archive[k] for k in archive.files}
        for i,record in enumerate(metrics):
            if record['model']!=model:continue
            key=(model,record['condition'],record['metric'],record['scope'],record['topic'])
            s.require(key not in seen,'duplicate endpoint');seen.add(key)
            definition=endpoints[(record['condition'],record['metric'])]
            old=scores['historical_'+record['condition']];new=scores['corrected_'+record['condition']]
            if record['condition']=='atomic_D':
                labels=np.array([int(r['label']) for r in atomic]);mask=np.ones(len(atomic),bool)
                eligible=atomic_keep;topics=np.array([r['topic'] for r in atomic])
            else:
                labels=raw.cell.isin(definition['positive']).to_numpy(int)
                mask=((raw.operator==definition['operator']) & raw.cell.isin(definition['positive']+definition['negative'])).to_numpy()
                eligible=compound_keep;topics=raw.topic.to_numpy()
            groups=[mask] if record['scope']=='pooled' else [mask & (topics==t) for t in
                    (config['topics'] if record['scope']=='topic_macro' else [record['topic']])]
            expected=np.mean([[roc_auc_score(labels[g],old[g]),roc_auc_score(labels[g & eligible],old[g & eligible]),
                               roc_auc_score(labels[g & eligible],new[g & eligible])] for g in groups],axis=0)
            actual=np.array([record[k] for k in ('historical_full_auroc','historical_retained_auroc','corrected_retained_auroc')])
            check_values(actual,expected);max_error=max(max_error,float(np.max(np.abs(actual-expected))))
            check_values([record['training_delta'],record['coverage_delta']], [expected[2]-expected[1],expected[1]-expected[0]])
            s.require(record['original_rows']==sum(int(g.sum()) for g in groups)
                      and record['retained_rows']==sum(int((g & eligible).sum()) for g in groups), 'coverage counts changed')
            for name,values in [('historical_ci',old_draws[:,i]),('corrected_ci',new_draws[:,i]),('paired_delta_ci',new_draws[:,i]-old_draws[:,i])]:
                valid=values[np.isfinite(values)];ci=record[name]
                s.require(len(valid)==ci['valid_replicates'] and len(valid)>=config['bootstrap']['minimum_valid'], 'uncertainty validity')
                check_values([ci['ci_low'],ci['ci_high']],np.quantile(valid,[.025,.975],method='linear'))
    refresh=json.loads((results/'refresh_decisions.json').read_text())
    for field,kind in [('training_delta','training_cleanup_auroc'),('coverage_delta','evaluation_coverage_auroc')]:
        expected={'/'.join(r[k] for k in ('model','condition','metric','scope')) for r in metrics
                  if r['scope'] in config['refresh']['summary_scopes'] and abs(r[field])>config['refresh']['absolute_auroc_movement_threshold']}
        actual={r['endpoint'] for r in refresh['triggers'] if r['kind']==kind}
        s.require(actual==expected,'refresh threshold mismatch')
    s.require(refresh['automatic_refresh_launched'] is False, 'unexpected broader refresh')
    return dict(status='passed',independent_auroc_implementation='sklearn.metrics.roc_auc_score',
                endpoint_summaries_checked=len(metrics),maximum_absolute_auroc_difference=max_error,
                all_result_and_external_artifact_hashes_match=True,all_intervals_recomputed=True,
                current_membership_and_refresh_thresholds_match=True,real_fits_performed=0,activation_arrays_loaded=0,
                sensitivity_receipt=s.identity((results/'sensitivity_receipt.json').read_bytes()),
                validator_source_sha256=s.sha(Path(__file__).read_bytes()))
