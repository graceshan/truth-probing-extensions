"""Same-support behavior and probe summaries; no thresholds, fitting or calibration."""
from collections import defaultdict
import math
from src.checkpoint_next_inventory import TOPICS, digest, require


def expand(x, records, prompt):
    require(len(records)==len({r['id'] for r in records}), 'duplicate physical result')
    lookup={r['id']:r for r in records}
    bs=[b for b in x['bindings'] if b['role']=='behavior' and b['prompt']==prompt]
    require({b['job_id'] for b in bs}<={*lookup}, 'missing physical behavior coverage')
    out=[]
    for b in bs:
        r=lookup[b['job_id']];l=r['likelihood'];g=r['generation']
        out.append(dict(b, margin=l['margin'],label_signed_margin=(2*b['label']-1)*l['margin'],
            likelihood_decision=l['decision'],generation_decision=g['decision'],format_failure=g['format_failure'],
            likelihood_correct=l['decision']==b['label'],generation_correct=g['decision']==b['label'],
            raw_output=g['raw_output'],token_ids=g['token_ids']))
    return out


def isolated_mask(rows):
    isolated=defaultdict(list)
    for r in rows:
        if r['group']=='behavior_isolated':isolated[r['pair_id']].append(r)
    require(len(isolated)==50 and all(len(v)==4 and len({r['fact_id'] for r in v})==4 for v in isolated.values()), 'mask coverage')
    return {p for p,rs in isolated.items() if all(r['likelihood_correct'] for r in rs)}


def ratio(n,d):
    return n/d if d else None


def rates(rows):
    n=len(rows);valid=[r for r in rows if not r['format_failure']];nv=len(valid)
    count=lambda field:sum(r[field] for r in rows)
    errors=sum(not r['generation_correct'] for r in valid)
    disagreement=sum(r['generation_decision']!=r['likelihood_decision'] for r in valid)
    return dict(rows=n,valid_generated=nv,likelihood_correct=count('likelihood_correct'),
        likelihood_accuracy=ratio(count('likelihood_correct'),n),generation_correct=count('generation_correct'),
        generation_accuracy=ratio(count('generation_correct'),n),format_failures=count('format_failure'),
        format_failure_rate=ratio(count('format_failure'),n),valid_factual_errors=errors,
        valid_factual_error_rate=ratio(errors,nv),disagreement=disagreement,disagreement_denominator=nv,
        disagreement_rate=ratio(disagreement,nv),likelihood_true=sum(r['likelihood_decision']==1 for r in rows),
        likelihood_true_rate=ratio(sum(r['likelihood_decision']==1 for r in rows),n),
        generation_true=sum(r['generation_decision']==1 for r in valid),
        generation_true_rate_all=ratio(sum(r['generation_decision']==1 for r in valid),n),
        generation_true_rate_valid=ratio(sum(r['generation_decision']==1 for r in valid),nv))


def summaries(rows, weights=None, options=None):
    import numpy as np
    from src.clean_transfer_statistics import Mean, interval
    records=[]
    lookup={r['binding_id']:i for i,r in enumerate(rows)}
    measures={'likelihood_accuracy':('likelihood_correct',False),'generation_accuracy':('generation_correct',False),
        'format_failure_rate':('format_failure',False),'valid_factual_error_rate':('factual_error',True),
        'disagreement_rate':('disagreement',True),'likelihood_true_rate':('likelihood_true',False),
        'generation_true_rate_all':('generation_true',False),'generation_true_rate_valid':('generation_true',True)}
    def value(r,field):
        if field=='factual_error':return not r['generation_correct']
        if field=='disagreement':return r['generation_decision']!=r['likelihood_decision']
        if field=='likelihood_true':return r['likelihood_decision']==1
        if field=='generation_true':return r['generation_decision']==1
        return r[field]
    for op in ('all','AND','OR'):
        for cell in ('all','TT','TF','FT','FF'):
            for order in ('all','AB','BA'):
                subset=[r for r in rows if (op=='all' or r['operator']==op) and
                    (cell=='all' or ''.join('T' if v=='1' else 'F' for v in (r['truth_a'],r['truth_b']))==cell) and
                    (order=='all' or r['ordering']==order)]
                scopes=[('pooled','all',subset)]+[('topic',t,[r for r in subset if r['topic']==t]) for t in TOPICS]
                topics=[];topic_draws={k:[] for k in measures}
                for scope,t,rs in scopes:
                    point=rates(rs);record=dict(operator=op,cell=cell,order=order,scope=scope,topic=t,**point)
                    if weights is not None:
                        for metric,(field,valid_only) in measures.items():
                            selected=[r for r in rs if not valid_only or not r['format_failure']]
                            indices=np.array([lookup[r['binding_id']] for r in selected],dtype=int)
                            draw=Mean(indices,np.array([value(r,field) for r in selected]))(weights)
                            record[metric+'_interval']=interval(draw,options)
                            if scope=='topic':topic_draws[metric].append(draw)
                    records.append(record)
                    if scope=='topic':topics.append(point)
                macro={k:None if any(v[k] is None for v in topics) else float(np.mean([v[k] for v in topics]))
                       for k in topics[0] if k.endswith('accuracy') or k.endswith('rate') or k.endswith('rate_all') or k.endswith('rate_valid')}
                record=dict(operator=op,cell=cell,order=order,scope='topic_macro',topic='all',
                    topic_denominators=[v['rows'] for v in topics],**macro)
                if weights is not None:
                    record.update({k+'_interval':interval(np.mean(v,axis=0),options) for k,v in topic_draws.items()})
                records.append(record)
    return records


def margin_summaries(rows):
    import numpy as np
    out=[]
    for topic in ('all',)+TOPICS:
        rs=[r for r in rows if r['group']=='behavior_isolated' and (topic=='all' or r['topic']==topic)]
        for label in ('all',0,1):
            selected=[r for r in rs if label=='all' or r['label']==label]
            for field in ('margin','label_signed_margin'):
                v=[r[field] for r in selected]
                out.append(dict(topic=topic,label=label,field=field,bindings=len(v),
                    unique_facts=len({r['fact_id'] for r in selected}),mean=float(np.mean(v)) if v else None,
                    quantiles=np.quantile(v,[0,.05,.25,.5,.75,.95,1]).tolist() if v else None,
                    mask_policy='correct sign only; no confidence filter'))
    return out


def aucs(rows, weights=None, options=None):
    """AND/OR/boundary only; no meaningless within-truth-cell AUROC."""
    import numpy as np
    from src.clean_transfer_statistics import AUC, interval
    y=np.array([r['label'] for r in rows]);s=np.array([r['margin'] for r in rows])
    defs={'AND':lambda r:r['operator']=='AND','OR':lambda r:r['operator']=='OR',
          'OR_boundary':lambda r:r['operator']=='OR' and int(r['truth_a'])+int(r['truth_b'])<2}
    records=[]
    for name,select in defs.items():
        topic_points=[];topic_draws=[]
        for scope,topic in [('pooled','all')]+[('topic',t) for t in TOPICS]:
            idx=np.array([i for i,r in enumerate(rows) if select(r) and (topic=='all' or r['topic']==topic)],dtype=int)
            fn=AUC(idx,s[idx],y[idx]);p=float(fn(np.ones((1,len(rows))))[0])
            rec=dict(metric=name+'_auroc',scope=scope,topic=topic,rows=len(idx),point=p if math.isfinite(p) else None)
            if weights is not None:
                draw=fn(weights);rec.update(interval(draw,options))
                if scope=='topic':topic_draws.append(draw)
            if scope=='topic':topic_points.append(p)
            records.append(rec)
        p=float(np.mean(topic_points));rec=dict(metric=name+'_auroc',scope='topic_macro',topic='all',point=p if math.isfinite(p) else None)
        if weights is not None:rec.update(interval(np.mean(topic_draws,axis=0),options))
        records.append(rec)
    return records


def joint_errors(rows, probe):
    """Require exact frozen logical support. AUROC-only arms have no decision table."""
    require(len(probe['logical_ids'])==len(set(probe['logical_ids'])),'duplicate probe row')
    lookup=dict(zip(probe['logical_ids'],probe['scores']))
    require(set(r['logical_id'] for r in rows)==set(lookup),'probe/behavior support mismatch')
    if probe['decision_rule'] is None:
        return dict(status='AUROC_only_no_frozen_decisions',rows=len(rows),joint_errors=None)
    require(probe['decision_rule']=='zero_logit_ties_true' and probe['frozen_rule_source_sha256'],'unsupported/unfrozen decision rule')
    require(all(math.isfinite(float(s)) for s in probe['scores']),'nonfinite probe scores')
    out=[]
    for mode in ('likelihood','generation'):
        grouped=defaultdict(lambda:dict(both_correct=0,probe_only_wrong=0,behavior_only_wrong=0,both_wrong=0,format_failures=0,valid_factual_errors=0,rows=0))
        for r in rows:
            key=(r['operator'],r['truth_a'],r['truth_b'],r['ordering'])
            v=grouped[key];pc=int(lookup[r['logical_id']]>=0)==r['label'];bc=r[mode+'_correct']
            category='both_correct' if pc and bc else 'probe_only_wrong' if bc else 'behavior_only_wrong' if pc else 'both_wrong'
            v[category]+=1;v['rows']+=1
            if mode=='generation':
                v['format_failures']+=r['format_failure'];v['valid_factual_errors']+=not r['format_failure'] and not bc
        out.extend(dict(mode=mode,operator=k[0],truth_a=k[1],truth_b=k[2],order=k[3],**v) for k,v in sorted(grouped.items()))
    return dict(status='frozen_decision_joint_errors',rows=len(rows),records=out,
                format_failure_interpretation='incorrect in primary accuracy; formatting separately flagged from valid factual error')


def match_raw_chat(raw, chat):
    for key in ('model','family','layer','fitting_identities_sha256','D_support_sha256'):
        require(raw[key]==chat[key],'raw/chat mismatch: '+key)
    require(raw['preprocessing_membership']=='original_P15' and chat['preprocessing_membership']=='original_P15', 'full-bank preprocessing contamination')
    require(raw['P15_identities_sha256']==chat['P15_identities_sha256'],'P15 identities differ')
    require(raw['regime']=='raw' and chat['regime']=='prompt1_chat','input regime mismatch')
    return dict(status='matched',head_family=raw['family'],layer=raw['layer'],support_sha256=raw['D_support_sha256'],
        interpretation='Only these Prompt 1 readouts; no prompted constituent availability or causal use established')


def report(x, physical_records, prompt, options=None):
    """Same draw schedule over full frozen support; secondary masks are held fixed."""
    import numpy as np
    import pandas as pd
    from src.selection_repair_canonical_sensitivity import compound_weights
    from src.clean_transfer_statistics import Mean, interval
    options=options or dict(replicates=2000,seed=1729,minimum_valid=1900)
    allrows=expand(x,physical_records,prompt);mask=isolated_mask(allrows)
    full=[r for r in allrows if r['group']=='behavior_compound']
    weights,schedule=compound_weights(pd.DataFrame(full).rename(columns={'person_a':'entity_a_id','person_b':'entity_b_id'}),np.ones(len(full),bool),options)
    populations={}
    for name,idx in [('full_primary',list(range(len(full)))),('isolated_correct_secondary',[i for i,r in enumerate(full) if r['pair_id'] in mask])]:
        rows=[full[i] for i in idx];w=weights[:,idx]
        acc=[]
        for field in ('likelihood_correct','generation_correct','format_failure'):
            topic_draws=[];points=[]
            for scope,t in [('pooled','all')]+[('topic',t) for t in TOPICS]:
                indices=np.array([i for i,r in enumerate(rows) if t=='all' or r['topic']==t],dtype=int)
                values=np.array([r[field] for r in rows])[indices]
                draw=Mean(indices,values)(w)
                point=float(np.mean(values)) if len(values) else None
                acc.append(dict(metric=field,scope=scope,topic=t,rows=len(indices),point=point,**interval(draw,options)))
                if scope=='topic':topic_draws.append(draw);points.append(point)
            acc.append(dict(metric=field,scope='topic_macro',topic='all',point=None if None in points else float(np.mean(points)),**interval(np.mean(topic_draws,axis=0),options)))
        populations[name]=dict(pairs=len({r['pair_id'] for r in rows}),rows=len(rows),
            topic_pair_coverage={t:len({r['pair_id'] for r in rows if r['topic']==t}) for t in TOPICS},
            accuracy=summaries(rows,w,options),accuracy_intervals=acc,ranking=aucs(rows,w,options))
    return dict(prompt=prompt,scope='frozen 50 pairs only; full-D probes reported separately',populations=populations,
        isolated_rates=isolated_rates(allrows),isolated_margins=margin_summaries(allrows),bootstrap=dict(options,schedule_sha256=schedule,
            unit='topic-stratified entity; endpoint multiplicity products',mask='fixed observed mask; no redraw',
            interval='pointwise conditional, undefined draws retained'),
        diagnosis='Interpret strong output-margin AUROC with weak decisions separately from valid factual errors and formatting failures; no D threshold fitting')


def probe_behavior_comparison(x, behavior_rows, probe, options=None):
    """Keep full-D ranking alongside the matched 800-row probe/behavior table."""
    require(probe['model'] in ('qwen','llama'),'probe model')
    import csv
    import numpy as np
    import pandas as pd
    from src.selection_repair_canonical_sensitivity import compound_weights
    from src.checkpoint_next_inventory import ROOT
    with (ROOT/'data/checkpoint_r2_v1/compound_bindings.csv').open() as f:
        full=[r for r in csv.DictReader(f) if r['group']=='D_bare']
    require(len(probe['logical_ids'])==len(probe['scores'])==len(set(probe['logical_ids'])),'probe score coverage')
    values=dict(zip(probe['logical_ids'],probe['scores']))
    require(set(values)=={r['example_id'] for r in full},'full-D probe coverage; no silent subset')
    full_scores=[dict(r,label=int(r['label']),margin=float(values[r['example_id']])) for r in full]
    matched=[dict(r,margin=float(values[r['logical_id']])) for r in behavior_rows]
    sub=dict(probe,logical_ids=[r['logical_id'] for r in behavior_rows],scores=[values[r['logical_id']] for r in behavior_rows])
    options=options or dict(replicates=2000,seed=1729,minimum_valid=1900)
    weights,schedule=compound_weights(pd.DataFrame(behavior_rows).rename(columns={'person_a':'entity_a_id','person_b':'entity_b_id'}),np.ones(len(behavior_rows),bool),options)
    return dict(head=probe['head_id'],full_D_probe_ranking=aucs(full_scores),
        same_50_pair_support_sha256=digest([r['logical_id'] for r in behavior_rows]),matched_probe_ranking=aucs(matched,weights,options),
        matched_behavior_ranking=aucs(behavior_rows,weights,options),joint=joint_errors(behavior_rows,sub),
        paired_entity_schedule_sha256=schedule,
        interpretation='Output-margin ranking and zero-margin decisions are distinct; formatting is not a factual error')


def isolated_rates(rows):
    isolated=[r for r in rows if r['group']=='behavior_isolated']
    require(len(isolated)==200 and len({r['fact_id'] for r in isolated})==150,'isolated logical support')
    return [dict(topic=t,weighting='logical bindings; no fact deduplication',
                 unique_facts=len({r['fact_id'] for r in rs}),**rates(rs))
            for t in ('all',)+TOPICS
            for rs in [[r for r in isolated if t=='all' or r['topic']==t]]]


def verified_chat_heads(x, directory, producer_sha, raw_heads):
    """Load exactly eight successful chat heads; enforce frozen identity and score axes.

    raw_heads are the reviewed raw export metadata plus full-D scores. They must
    carry original P15 fitting identities and frozen decision-rule provenance.
    """
    import json
    import numpy as np
    from pathlib import Path
    from src.checkpoint_next_inventory import sha
    root=Path(directory);completion=json.loads((root/'completion.json').read_text())
    expected={h['id']:h for h in x['heads']}
    require(len(raw_heads)==8 and {r['id'] for r in raw_heads}==set(expected),'eight exact raw comparator identities')
    raw={r['id']:r for r in raw_heads}
    require(completion['status']=='completed' and completion['producer_commit']==producer_sha and
        len(completion['heads'])==8 and {h['id'] for h in completion['heads']}==set(expected),'eight chat completion identities')
    output=[]
    groups={g:[r for r in x['bindings'] if r['role']=='chat' and r['group']==g]
            for g in ('P15','TC_control_and_final_refit','behavior_D')}
    for item in completion['heads']:
        head=expected[item['id']];folder=root/head['id'];receipt=json.loads((folder/'receipt.json').read_text())
        require(sha(folder/'receipt.json')==item['receipt_sha256'],'chat receipt hash')
        require(receipt['raw_matches_content_sha256']==digest(raw_heads),'raw comparator export changed since fit')
        require(receipt['producer_commit']==producer_sha and receipt['status']=='converged' and
                receipt['prediction_usable'] is True,'failed/wrong producer chat fit')
        for key,value in head.items():require(receipt['head'][key]==value,'frozen chat head identity: '+key)
        require(receipt['preprocessing_source']=='original chat-P15 only' and receipt['objective']==head['objective'],'chat objective/preprocessing')
        require(receipt['head']['P15_identities_sha256']==digest([r['logical_id'] for r in groups['P15']]),'chat P15 identities')
        for filename,key in [('parameters.npz','parameters_sha256'),('scores.npz','scores_sha256'),('fit-status.json','fit_status_sha256')]:
            require(sha(folder/filename)==receipt[key],'chat artifact hash: '+filename)
        status=json.loads((folder/'fit-status.json').read_text())
        require(all(receipt[k]==v for k,v in status.items()),'fit status receipt mismatch')
        require(receipt['verification']['identity']['producer_commit']==producer_sha and
                receipt['verification']['identity']['inventory_sha256']==digest(x) and
                receipt['verification']['model']==head['model'] and receipt['verification']['status']=='verified','chat extraction provenance')
        identities={g:[r['logical_id'] for r in rs] for g,rs in groups.items()}
        require(receipt['score_identities']==identities,'chat exact logical score order')
        match=match_raw_chat(raw[head['id']],receipt['head'])
        require(raw[head['id']]['decision_rule']==head['decision_rule'] and raw[head['id']]['frozen_rule_source_sha256'], 'frozen matching decision rule')
        with np.load(folder/'scores.npz',allow_pickle=False) as scores:
            require(set(scores.files)==set(groups),'chat score groups')
            for g in groups:require(scores[g].shape==(len(groups[g]),) and np.isfinite(scores[g]).all(),'chat score axes/finite')
            d=scores['behavior_D'].tolist()
        require(len(d)==800 and digest(identities['behavior_D'])==head['D_support_sha256'],'chat 800 frozen support')
        output.append(dict(receipt['head'],head_id=head['id'],logical_ids=identities['behavior_D'],scores=d,
            frozen_rule_source_sha256=raw[head['id']]['frozen_rule_source_sha256'],match=match,receipt_sha256=item['receipt_sha256']))
    return output


def chat_behavior_comparison(behavior_rows, chat, raw):
    """Chat has only 800 D scores. Full-D raw remains a separate report arm."""
    require(len(behavior_rows)==800 and len({r['pair_id'] for r in behavior_rows})==50 and
            all(r['prompt']==1 for r in behavior_rows),'chat Prompt 1 support')
    match=match_raw_chat(raw,chat)
    ids=[r['logical_id'] for r in behavior_rows]
    require(len(chat['logical_ids'])==800 and set(chat['logical_ids'])==set(ids),'chat support')
    import csv
    from src.checkpoint_next_inventory import ROOT
    with (ROOT/'data/checkpoint_r2_v1/compound_bindings.csv').open() as f:
        full=[r['example_id'] for r in csv.DictReader(f) if r['group']=='D_bare']
    require(len(raw['logical_ids'])==len(raw['scores'])==len(set(raw['logical_ids']))==7728 and
            set(raw['logical_ids'])==set(full),'raw full-D coverage')
    rv=dict(zip(raw['logical_ids'],raw['scores']));cv=dict(zip(chat['logical_ids'],chat['scores']))
    require(all(math.isfinite(float(v)) for v in rv.values()),'raw finite scores')
    raw_sub=dict(raw,logical_ids=ids,scores=[rv[i] for i in ids])
    return dict(head=chat['head_id'],matching=match,rows=800,pairs=50,
        raw_ranking=aucs([dict(r,margin=rv[r['logical_id']]) for r in behavior_rows]),
        chat_ranking=aucs([dict(r,margin=cv[r['logical_id']]) for r in behavior_rows]),
        raw_joint=joint_errors(behavior_rows,raw_sub),chat_joint=joint_errors(behavior_rows,chat),
        full_D_chat_status='not collected; no full-D chat estimate',
        interpretation='Named Prompt 1 readouts only; no prompted constituent availability or causal use')
