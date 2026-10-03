"""Exactly eight chat recipes using the reviewed P-only preprocessing/objectives.

prepare_blocks is pure input preparation. fit_prepared is a separately gated future
CPU entry point, called only after all extracted inputs and raw matches are verified.
"""
from src.checkpoint_next_inventory import digest, require


def prepare_blocks(head, p_rows, p_features, tc_rows, tc_features):
    from src.selection_repair_objectives import fit_p_preprocessing, prepare_block
    require(head['family'] in ('R0','C_clean') and head['preprocess_group']=='P15', 'chat fitting scope')
    require(len(p_rows)==2778 and len(tc_rows)==8000,'original fitting counts')
    require(all(r['group']=='P15' for r in p_rows) and all(r['group']=='TC_control_and_final_refit' for r in tc_rows), 'full-train contamination')
    require(digest(p_rows)==head['preprocess_bindings_sha256'], 'chat-P15 binding order')
    fitting=p_rows if head['family']=='R0' else tc_rows
    require(digest([r['logical_id'] for r in fitting])==head['fitting_identities_sha256'],'fitting identities changed')
    pre=fit_p_preprocessing(p_features)
    # Fixed TC has all 16 leaves per pair, so its equal-pair/cell/order objective
    # equals the existing ordinary 8,000-row mean. Require that equivalence explicitly.
    if head['family']=='C_clean':
        from src.selection_repair_objectives import compound_pair_weights
        import numpy as np
        weights=compound_pair_weights([r['pair_id'] for r in tc_rows],[r['operator'] for r in tc_rows],
            [(int(r['truth_a']),int(r['truth_b'])) for r in tc_rows],[0 if r['ordering']=='AB' else 1 for r in tc_rows])
        require(np.allclose(weights,1/8000,rtol=0,atol=1e-15),'TC weighting equivalence')
        block=prepare_block(tc_features,[r['label'] for r in tc_rows],pre,weights)
    else:
        block=prepare_block(p_features,[r['label'] for r in p_rows],pre)
    return pre,block


def fit_prepared(head, pre, block, authorization, producer_sha):
    from src.selection_repair_objectives import fit_readout
    require(authorization['phase']=='chat_cpu_fit' and authorization['reviewed'] is True and
            authorization['producer_commit']==producer_sha and authorization['heads']==8 and
            authorization['CPU_slot_exclusive'] is True, 'separate reviewed CPU launch/slot required')
    require(pre.p_rows==2778,'P15 preprocessing count')
    # r0 API is the existing single-block mean-BCE + .001 penalty objective;
    # passing fixed TC here implements existing C_clean, never the mixed repair loss.
    fit=fit_readout('r0',block,pre)
    require(fit.converged,'optimizer failure blocks affected chat task')
    return fit
