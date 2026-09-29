"""Fabricated in temporary directories; never read scientific datasets."""
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from src import clean_transfer_contracts as c
from src import pinned_compound_scoring as scoring
from src.clean_extraction import layer_convention
from src.pinned_atomic_probes import RepairedAtomicData
from test_pinned_atomic_probes import PinnedAtomicTests


def graph(sizes=None):
    sizes = sizes or dict(zip(c.TOPICS, [5, 6, 7, 8, 9]))
    records = []
    for topic, n in sizes.items():
        pairs = sorted({tuple(sorted([i, (i+d) % n])) for i in range(n) for d in [1, 2]})
        for i, j in pairs:
            a, b, pair = f'{topic}_entity_{i}', f'{topic}_entity_{j}', f'{topic}_pair_{i}_{j}'
            for op, ta, tb, order in product(['AND', 'OR'], [True, False], [True, False], ['AB', 'BA']):
                records.append(dict(example_id=f'{pair}_{op}_{ta}_{tb}_{order}', pair_id=pair, topic=topic,
                    split='validation', protocol='entity_disjoint', evaluation_phase='development',
                    benchmark_label='DEVELOPMENT / VALIDATION - NOT FINAL TEST',
                    statement=f'Synthetic {a} {ta} {op} {b} {tb} {order}.', operator=op, ordering=order,
                    entity_a_id=a, entity_b_id=b, fact_a_id=f'{a}_{ta}', fact_b_id=f'{b}_{tb}',
                    canonical_truth_a=ta, canonical_truth_b=tb,
                    surface_first_truth=ta if order == 'AB' else tb, surface_second_truth=tb if order == 'AB' else ta,
                    compound_label=(ta and tb) if op == 'AND' else (ta or tb)))
    frame = pd.DataFrame(records).sort_values('example_id').reset_index(drop=True)
    expected = {**c.BENCHMARK, 'rows': len(frame), 'entities': sum(sizes.values()), 'pairs': 2*sum(sizes.values()),
                'pairs_per_topic': {t: 2*n for t, n in sizes.items()}}
    return frame, expected


def rewrite(path, value):
    path.write_bytes(c.canonical(value))


@pytest.fixture
def artifacts(monkeypatch):
    fixture = PinnedAtomicTests()
    fixture.setUp()
    try:
        root = fixture.root
        frame, benchmark = graph()
        monkeypatch.setattr(c, 'WIDTH', 3)
        monkeypatch.setattr(c, 'ROWS', len(frame))
        monkeypatch.setattr(c, 'BENCHMARK', benchmark)
        fixture.manifest['resolved_model'].update(config_file_sha256='c'*64,
            tokenizer_config_file_sha256='d'*64, tokenizer_backend_sha256='e'*64)
        fixture.refresh_receipt()
        data = RepairedAtomicData(root)
        directory = root / c.PROBE
        directory.mkdir(parents=True)
        coef = np.array([[.3, -.2, .7]])
        np.savez_compressed(directory / 'selected_probe.npz', coef=coef, intercept=np.array([.1]), classes=np.array([0, 1]),
                            layer=np.array(17), C=np.array(1.), n_iter=np.array([12]), max_iter=np.array(2000))
        grid = []
        for layer, C in product(range(28), c.C_VALUES):
            grid.append(dict(layer=layer, C=C, validation_overall_auroc=(c.EXPECTED_SELECTION['validation_auroc']
                        if (layer, C) == (17, 1.) else .6), final_converged=True, final_convergence_status='converged',
                        convergence_warning=False, final_max_iter=2000, final_n_iter=12))
        pd.DataFrame(grid).to_csv(directory / 'validation_metrics.csv', index=False, float_format='%.17g')
        structure = dict(data.structural)
        structure['adapter_code_sha256'] = {'src/pinned_atomic_probes.py': 'a'*64}
        selection = dict(model=c.MODEL, selected_layer=17, selected_C=1., selection_rule=c.SELECTION_RULE,
            C_values=c.C_VALUES, probe_configuration=scoring.PROBE_CONFIG, preprocessing='none',
            label_convention='0=false, 1=true; score >= 0 predicts true',
            layer_convention='zero-based transformer-block output; embedding excluded',
            structural=structure, provenance=dict(config=data.config, config_sha256=c.digest(c.canonical(data.config)),
                model_key='qwen25_7b', code_sha256={}, git_revision=None),
            selected_probe_sha256=c.file_hash(directory / 'selected_probe.npz'),
            validation_metrics_sha256=c.file_hash(directory / 'validation_metrics.csv'), test_evaluated=False,
            test_decision_scores_computed=0, activation_rows_read={'test': 0, 'train': 560, 'validation': 560},
            all_final_fits_converged=True, selected_probe_max_iter=2000,
            selected_validation_metrics=next(r for r in grid if r['layer'] == 17 and r['C'] == 1.))
        # Production selection serialization sorts the document keys, but hashes
        # the original insertion-order config. Reproduce that original order via
        # the producer's config shape in the verifier (covered by the tests).
        selection['provenance']['config_sha256'] = c.digest(json.dumps(data.config, ensure_ascii=False, separators=(',', ':')).encode())
        (directory / 'selection.json').write_text(json.dumps(selection))
        compound = root / c.COMPOUND
        compound.mkdir(parents=True)
        frame.to_csv(compound / 'metadata.csv', index=False)
        values = np.random.default_rng(32).normal(size=(len(frame), 28, 3)).astype(np.float16)
        np.save(compound / 'activations.npy', values)
        monkeypatch.setattr(c, 'ACTIVATION_SHA', c.file_hash(compound / 'activations.npy'))
        monkeypatch.setattr(c, 'METADATA_SHA', c.file_hash(compound / 'metadata.csv'))
        descriptor = c.representation(data.manifest['contract'], layer_convention(28))
        monkeypatch.setattr(c, 'FINGERPRINT', c.digest(c.canonical(descriptor)))
        binding = dict(passed=True, repaired_cache_files=data.files, repaired_cache_identity_sha256=c.digest(c.canonical(data.files)),
                       representation=descriptor, representation_fingerprint=c.FINGERPRINT)
        manifest = dict(representation=descriptor, representation_fingerprint=c.FINGERPRINT,
            execution_contract=data.manifest['contract'], resolved_model=data.manifest['resolved_model'],
            repaired_atomic_binding=binding, atomic_replay_sample_sha256='b'*64,
            smoke_test=dict(live_binding_passed=True,
                atomic_replay=dict(passed=True, saved_float16_byte_equal=True, sample_sha256='b'*64, batch_size=1,
                                   padding=False, policy='exact-repaired-atomic-float16-replay-v1'),
                compound_smoke=dict(passed=True, batch_size=1, padding=False, policy='unpadded-single-repeat-direct-exact-v1')),
            code=dict(implementation_version='synthetic', source_sha256={}),
            data=dict(expected_activation_shape=list(values.shape), number_of_examples=len(frame),
                      sidecar_sha256=c.METADATA_SHA, benchmark_sha256=c.METADATA_SHA,
                      ordered_example_id_sha256=c.ordered_hash(frame.example_id)))
        rewrite(compound / 'extraction_manifest.json', manifest)
        progress = dict(complete=True, next_row=len(frame), activation_file_sha256=c.ACTIVATION_SHA,
                        identity_sha256=scoring.extraction_identity(manifest), chunks=[dict(start=0, stop=len(frame),
                        ordered_example_id_sha256=c.ordered_hash(frame.example_id), activation_sha256=c.digest(values.tobytes()))])
        rewrite(compound / 'progress.json', progress)
        yield root, frame, values
    finally:
        fixture.doCleanups()
