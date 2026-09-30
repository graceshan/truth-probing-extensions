"""TRAIN/VALIDATION-only Llama adapter to the unchanged clean LR selector."""
import io
import numpy as np
import pandas as pd

from src import clean_transfer_contracts as c
from src import clean_atomic_probes as selector
from src import llama_replication_contracts as l
from src import llama_replication_extraction as extraction


class AtomicData:
    _authorize_test = False

    def __init__(self, root=c.ROOT):
        self.root = root
        self.pin = l.read_pin(root)
        self.groups, identity = extraction.atomic_rows(root)
        self.files = {split: extraction.verify_cache(root, split, rows, identity) for split, rows in self.groups.items()}
        self.n_layers, self.hidden_size = l.LAYERS, l.WIDTH
        self.model_key = 'llama31_8b_instruct'
        self.model = dict(model=l.MODEL, n_layers=l.LAYERS, hidden_size=l.WIDTH, dtype='float16')
        self.config = dict(loader='llama31-train-validation-only-v1', C_values=selector.C_VALUES, model=self.model)
        self.counts = [dict(split=split, rows=len(rows.frame)) for split, rows in self.groups.items()]
        self.structural = dict(pin=self.pin, cache_files=self.files, approved_atomic_input=identity,
            counts=self.counts, training_policy='train-only fit; validation-only layer/C selection; no refit',
            test_accessed=False, compound_accessed=False)
        self.activation_rows_read = dict(train=0, validation=0, test=0)

    def partition(self, split):
        if split not in ['train', 'validation']:
            raise PermissionError('only TRAIN/VALIDATION exposed')
        return Partition(self, split)


class Partition:
    def __init__(self, data, split):
        self.data, self.split = data, split
        self.n_layers, self.hidden_size = data.n_layers, data.hidden_size
        frame = data.groups[split].frame
        self.entity_ids = frozenset(frame.entity_id)
        self.labels = frame.label.to_numpy(np.int64)
        self.topics, self.forms = frame.topic.to_numpy(), frame.form.to_numpy()

    def matrix(self, layer):
        c.require(self.split in ['train', 'validation'] and 0 <= layer < self.n_layers, 'invalid atomic slice')
        array = np.load(extraction.cache_path(self.data.root, self.split)/'activations.npy', mmap_mode='r', allow_pickle=False)
        values = np.asarray(array[:, layer, :], dtype=np.float64)
        c.require(np.isfinite(values).all(), 'nonfinite atomic activations')
        self.data.activation_rows_read[self.split] += len(values)
        return values


def select(root=c.ROOT, *, check_only=False):
    data = AtomicData(root)
    if check_only:
        return dict(structural=data.structural, fits_performed=0, activation_rows_read=data.activation_rows_read)
    output = c.safe_path(root, l.PROBE)
    output.mkdir(parents=True, exist_ok=False)
    result = selector.fit_and_save(data, output)
    c.publish_json(output/'completion.json', dict(complete=True, files={name: c.record(output/name)
        for name in ['selection.json', 'selected_probe.npz', 'validation_metrics.csv', 'split_counts.csv']},
        fits=l.LAYERS*len(selector.C_VALUES), test_accessed=False, compound_accessed=False))
    return result


def verify_probe(root=c.ROOT):
    """Read-only verification; no fits, compound reads or test partition."""
    data = AtomicData(root)
    path = c.safe_path(root, l.PROBE)
    receipt = c.read_json(c.safe_path(root, l.PROBE+'/completion.json'))
    names = ['selection.json', 'selected_probe.npz', 'validation_metrics.csv', 'split_counts.csv']
    c.require(receipt == dict(complete=True, files={name: c.record(c.safe_path(root, l.PROBE+'/'+name)) for name in names},
        fits=l.LAYERS*len(selector.C_VALUES), test_accessed=False, compound_accessed=False), 'probe completion/hash mismatch')
    selection = c.read_json(path/'selection.json')
    c.require(selection['model'] == l.MODEL and selection['structural'] == data.structural and
              selection['preprocessing'] == 'none' and selection['probe_configuration'] == selector.PROBE_CONFIG and
              selection['optimization_policy'] == selector.OPTIMIZATION_POLICY and
              selection['C_values'] == selector.C_VALUES and selection['selection_rule'] == selector.SELECTION_RULE and
              selection['label_convention'] == '0=false, 1=true; score >= 0 predicts true' and
              selection['test_evaluated'] is False and selection['test_decision_scores_computed'] == 0 and
              selection['activation_rows_read']['test'] == 0 and selection['all_final_fits_converged'] is True,
              'selection semantics/binding mismatch')
    c.require(selection['selected_probe_sha256'] == receipt['files']['selected_probe.npz']['sha256'] and
              selection['validation_metrics_sha256'] == receipt['files']['validation_metrics.csv']['sha256'], 'probe selection hashes')
    metrics = pd.read_csv(path/'validation_metrics.csv', float_precision='round_trip')
    c.require(len(metrics) == l.LAYERS*len(selector.C_VALUES) and
              set(zip(metrics.layer, metrics.C)) == {(layer, C) for layer in range(l.LAYERS) for C in selector.C_VALUES} and
              metrics.final_converged.eq(True).all() and metrics.convergence_warning.eq(False).all() and
              metrics.validation_overall_auroc.between(0, 1).all() and
              metrics.final_convergence_status.eq('converged').all() and
              (metrics.final_n_iter > 0).all() and (metrics.final_n_iter <= metrics.final_max_iter).all(),
              'complete converged validation grid required')
    winner = min(metrics.to_dict('records'), key=selector.selection_key)
    c.require(winner['layer'] == selection['selected_layer'] and winner['C'] == selection['selected_C'] and
              winner['validation_overall_auroc'] == selection['selected_validation_metrics']['validation_overall_auroc'],
              'validation-only winner mismatch')
    for key, value in selection['selected_validation_metrics'].items():
        c.require(key in winner and winner[key] == value, 'selected validation record mismatch: '+key)
    with np.load(io.BytesIO((path/'selected_probe.npz').read_bytes()), allow_pickle=False) as archive:
        c.require(set(archive.files) == {'coef', 'intercept', 'classes', 'layer', 'C', 'n_iter', 'max_iter'},
                  'probe archive schema')
        coef, intercept = archive['coef'].copy(), archive['intercept'].copy()
        c.require(int(archive['layer']) == winner['layer'] and float(archive['C']) == winner['C'] and
                  np.array_equal(archive['classes'], [0, 1]) and coef.shape == (1, l.WIDTH) and intercept.shape == (1,) and
                  np.isfinite(coef).all() and np.isfinite(intercept).all(), 'probe parameter contract')
    return selection, coef, intercept, data.structural
