"""Preparation schemas: explicit unrun rows, separate heads/exposures and curves."""
import math
from src.checkpoint_r2_selection_v1 import ENDPOINTS, require

FIELDS = ('panel', 'model', 'correction_version', 'representation_contract', 'fit_identity',
          'fitting_exposure', 'sample_seed', 'layer', 'method', 'evaluation_condition',
          'source_operator', 'target_operator', 'fit_stage', 'selection_binding',
          'evaluation_binding', 'row_count', 'entity_count', 'weights_binding',
          'endpoint', 'scope', 'topic', 'metric_value', 'ci_low', 'ci_high',
          'valid_replicates', 'undefined_replicates', 'draw_binding', 'status', 'reason')
ARMS = ('S_all','R_all','S_atom_all','original_atomic_selected_R0','reduced_LR',
        'S_fixed','R_fixed','LR_only_selection','repair_at_S_all','R0_fixed','R0_at_R_all',
        'atomic_only_fixed','atomic_only_at_R_all','bank_oracle','C_clean_fixed',
        'C_clean_D_selected','C_clean_at_S_all','C_clean_at_R_all','C_clean_at_S_atom_all')


def row(**values):
    result = dict.fromkeys(FIELDS)
    result.update(status='unrun', correction_version='pending_v5_identity_equivalence',
                  reason='Preparation only; common checkpoint and compatible input bindings pending')
    result.update(values)
    return result


def validate(row):
    require(set(FIELDS) <= set(row), 'missing result schema fields')
    require(row['status'] in ('unrun','failed','undefined','computed'), 'invalid result status')
    if row['status'] != 'computed':
        require(bool(row['reason']) and row['metric_value'] is None, 'missing failure reason or invented result')
    else:
        require(row['metric_value'] is not None and math.isfinite(row['metric_value']), 'nonfinite computed metric')
        require(all(row[k] for k in ('representation_contract','fit_identity','fitting_exposure','correction_version','evaluation_binding','weights_binding')),
                'computed result lacks fit/input identity')
    if row['panel'] in ('constituent_transfer','constituent_layer_curves','source_validation_curves'):
        require(row['endpoint'] in ENDPOINTS, 'six-endpoint vector required')
        require(row['fit_stage'] in ('source_fitted_16_per_topic','full_TC_refitted_20_per_topic'), 'ambiguous constituent fit stage')
        if row['panel'] == 'source_validation_curves':
            require(row['evaluation_condition']=='TC_source_validation' and row['source_operator']==row['target_operator']
                    and row['fit_stage']=='source_fitted_16_per_topic', 'target/full-refit leaked into source validation')
    if row['sample_seed'] == 'mean_11_23_37':
        require(row.get('aggregation') == 'mean_sample_metrics_per_shared_draw', 'pooled heads forbidden')
    if row['ci_low'] is not None or row['ci_high'] is not None:
        require(row['status']=='computed' and all(row[k] is not None for k in ('ci_low','ci_high','valid_replicates','undefined_replicates','draw_binding')),
                'interval lacks paired draw provenance')
        require(row['valid_replicates']+row['undefined_replicates']==2000, 'incorrect bootstrap count')
    return True


def templates():
    result = []
    for model in ('qwen','llama'):
        for arm in ('S_atom_all','original_atomic_selected_R0','reduced_LR','bank_oracle','C_clean_fixed','C_clean_D_selected','C_clean_at_S_all','C_clean_at_R_all','C_clean_at_S_atom_all'):
            result.append(row(panel='recoverability',model=model,method=arm,endpoint='all_declared_global_endpoints',evaluation_condition='bare_D_and_wording',scope='pooled_macro_and_each_topic'))
        for arm in ARMS:
            for seed in (11,23,37,'mean_11_23_37'):
                result.append(row(panel='b25_contrasts', model=model, method=arm, sample_seed=seed,
                                  endpoint='or_mixed_vs_ff_auroc', scope='pooled', topic='all',
                                  evaluation_condition='bare_D', aggregation='mean_sample_metrics_per_shared_draw' if isinstance(seed,str) else 'sample_specific'))
        for source, layer_origin in [('AND','AND_source_selected'),('OR','OR_source_selected'),
                                     ('OR','AND_source_selected'),('AND','OR_source_selected')]:
            for target in ('AND','OR'):
                for endpoint in ENDPOINTS:
                    result.append(row(panel='constituent_transfer',model=model,source_operator=source,
                                      target_operator=target,layer=layer_origin,endpoint=endpoint,
                                      evaluation_condition='bare_D_and_each_declared_wording',fit_stage='full_TC_refitted_20_per_topic'))
        for layer in range(28 if model=='qwen' else 32):
            for source in ('AND','OR'):
                for endpoint in ENDPOINTS:
                    result.append(row(panel='source_validation_curves',model=model,layer=layer,source_operator=source,
                                      target_operator=source,endpoint=endpoint,evaluation_condition='TC_source_validation',
                                      fit_stage='source_fitted_16_per_topic'))
                    for target in ('AND','OR'):
                        result.append(row(panel='constituent_layer_curves',model=model,layer=layer,source_operator=source,
                                          target_operator=target,endpoint=endpoint,evaluation_condition='bare_D_and_each_declared_wording',
                                          fit_stage='source_fitted_16_per_topic'))
    for r in result:
        validate(r)
    return result


SCHEMA = dict(schema_version=1, fields=list(FIELDS), statuses=['unrun','failed','undefined','computed'],
              required_panels=['recoverability','b25_contrasts','constituent_transfer','constituent_layer_curves','source_validation_curves','behavior_pilot','behavior_isolated_facts','validity_and_cost'],
              required_packet_files=['checkpoint_summary.md','recoverability.csv','b25_contrasts.csv','constituent_transfer.csv','constituent_layer_curves.csv','source_validation_curves.csv','behavior_pilot.csv','behavior_isolated_facts.csv','exposure_manifest.json','fit_status.csv','bridge_receipt.json','decision_record.md'],
              required_scopes=['pooled','equal_topic_macro','per_topic'],
              global_endpoints=['atomic_auroc','and_auroc','or_auroc','and_tt_vs_mixed_auroc','or_mixed_vs_ff_auroc'],
              constituent_endpoints=list(ENDPOINTS),
              constituent_additional=['surface-order-specific six endpoints','frozen .5 decisions: accuracy/balanced accuracy','joint correctness','AND min logits / OR max logits AUROC','Boolean decision composition','source-D and target-D absolute competence','same-layer target-trained minus source-trained paired six-endpoint losses'],
              primary_B25_contrasts=['R_all_minus_S_all','S_all_minus_S_atom_all','repair_minus_atomic_only','repair_minus_same_layer_R0','C_clean_minus_selection','atomic_AND_retention','cross_model_difference_of_repair_effects'],
              procedure_controls='C_clean fit identity is model/layer/TC/P-preprocessing/representation; resolve S_all, R_all and S_atom_all layer references onto existing C_clean heads, deduplicate',
              curve_separation='All-layer curves use source-fitted 16/topic heads. Selected/matched full-TC-refit results use 20/topic heads. Never overwrite or mix their fit IDs.',
              sample_aggregation='separate sample metrics then mean, including inside each paired draw; never concatenate separately fitted head scores',
              original_controls=list(ARMS),
              uncertainty='shared topic-stratified person/entity draws; endpoint products for compounds; seed1729/2000; preserve undefined topics/draws; pointwise conditional intervals',
              behavior_status='schema only, not implemented or authorized here',
              behavior_fields=['model','prompt','full_or_isolated_correct_mask','frozen_mask_hash','pair_entity_topic_coverage','isolated_fact_binding','likelihood_decision','generation_decision','canonical_format_valid','disagreement_valid_answer_denominator','format_failure_all_output_denominator'],
              planned_vs_implemented='These templates are unrun obligations; selector primitives and metric averaging are implemented, production result assembly is pending')
