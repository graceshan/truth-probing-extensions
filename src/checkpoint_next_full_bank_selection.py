"""Frozen full-bank selectors and paired reporting contracts; no objectives or fits."""
import numpy as np
from src.checkpoint_next_inventory import ROOT, require
from src.checkpoint_r2_fresh_inputs import hash_value
from src.checkpoint_r2_cpu_preflight_v1 import grid
from src.checkpoint_r2_b25_selection import rank
from src.checkpoint_r2_selection_v1 import AtomicCandidate, atomic_eligibility, select_atom_all, average_sample_metrics, TOPICS
from src.checkpoint_next_full_bank_inputs import read

CONTRASTS=(('R_all','S_full'),('S_full','S_atom_full'),('R_all','S_atom_full'),('S_full','S_all'))


def bank_specs():
    old=grid(read(ROOT/'config/clean_protocol/checkpoint_r2_cpu_preflight_v1.json'))
    return [dict(canonical_id=r['candidate_id'],fit_id=r['candidate_id'].replace('/P15/','/full_train/'),
        model=r['model'],saved_layer=r['saved_layer'],method=r['method'],C=r['C'],
        training_group='balanced_full' if r['method'] in ('burger_t_g','ttpd') else 'outer_train',
        fit_rows=1000 if r['method'] in ('burger_t_g','ttpd') else 3040,
        preprocessing_group='outer_train' if r['method']=='r0' else 'existing_method_specific_raw',
        research_status='unrun') for r in old]


def selectors(records,model,reference_atomic,reference_id):
    """Keep the original canonical recipe IDs for exact inherited randomized ties.

    Mapping to full-training fit IDs is explicit; no P15 fit/parameters are reused.
    Eligibility always uses the ORIGINAL reduced-training LR atomic D reference.
    """
    specs=[s for s in bank_specs() if s['model']==model]
    require(len(records)==len(specs) and {r['spec']['canonical_id'] for r in records}=={s['canonical_id'] for s in specs},'complete per-model bank before selection')
    require(all(r['valid_for_scoring'] for r in records),'failed candidate blocks complete-bank selectors')
    candidates=[AtomicCandidate(r['spec']['canonical_id'],model,r['spec']['saved_layer'],r['spec']['method'],r['spec']['C'],True,
        r['atomic']['pooled'],r['atomic']['topics']) for r in records]
    eligibility=atomic_eligibility(candidates,model,reference_atomic);allowed=[r['candidate_id'] for r in eligibility if r['eligible']]
    atom=select_atom_all(candidates,model=model,reduced_lr_atomic_auc=reference_atomic,expected_s_all_eligible_ids=allowed)
    require(atom['selected_id'] is not None,'atomic-only selector unresolved')
    mapping={r['spec']['canonical_id']:r['spec']['fit_id'] for r in records}
    traces={};arms={}
    for seed in (11,23,37):
        selected=rank([dict(id=r['spec']['canonical_id'],valid_for_scoring=True,**r['allocated'][str(seed)]) for r in records],
            reference_atomic=reference_atomic,fallback_id=reference_id,arm='S_full',eligible_ids=allowed)
        traces[str(seed)]=selected;arms[str(seed)]=dict(S_full=mapping.get(selected['selected_id'],reference_id),S_atom_full=mapping[atom['selected_id']])
    oracle={}
    for label,ids in [('eligible',set(allowed)),('unrestricted',set(mapping))]:
        eligible=[r for r in records if r['spec']['canonical_id'] in ids]
        require(eligible,'empty descriptive bank reference')
        # Existing B25 oracle uses the same primary/atomic/separation/tie ranking,
        # and eligible maximum shares the ORIGINAL retention reference.
        if label=='eligible':
            trace=rank([dict(id=r['spec']['canonical_id'],valid_for_scoring=True,**r['oracle']) for r in eligible],
                reference_atomic=reference_atomic,fallback_id=reference_id,arm='full_bank_oracle',eligible_ids=allowed)
            winner=trace['selected_id']
        else:
            best=max(r['oracle']['primary'] for r in eligible)
            tied=[r for r in eligible if best-r['oracle']['primary']<=1e-12]
            atomic=max(r['oracle']['atomic'] for r in tied);tied=[r for r in tied if atomic-r['oracle']['atomic']<=1e-12]
            sep=max(r['oracle']['separation'] for r in tied);ids=sorted(r['spec']['canonical_id'] for r in tied if sep-r['oracle']['separation']<=1e-12)
            winner=str(np.random.Generator(np.random.PCG64(20261008)).choice(ids))
            trace=dict(arm='full_bank_unrestricted_max',selected_id=winner,retention_gate=False,descriptive=True)
        oracle[label]=dict(fit_id=mapping.get(winner,reference_id),trace=trace,fixed_winner_conditional_intervals=True)
    return dict(model=model,reference_atomic=reference_atomic,reference_id=reference_id,eligibility=eligibility,
        atomic_only=atom,compound=traces,procedures=arms,oracle=oracle,wording_used=False,
        once_per_model_atomic_selector=True,A_identities_in_atomic_training=True)


def average_contrast(left,right):
    """Already computed metrics per seed/draw only. NaNs propagate, no surviving mean."""
    return average_sample_metrics({s:np.asarray(left[s])-np.asarray(right[s]) for s in (11,23,37)})
