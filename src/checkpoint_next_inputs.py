"""Read-only resolution of October 3 saved D scores and accepted fresh metadata."""
import json
import subprocess
from pathlib import Path
import numpy as np
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_constituent_adapter import ConstituentAdapter, check_production_packet

CONFIG = ROOT / 'config/checkpoint_r2/next_score_diagnosis_20261003.json'
OR_GROUPS = ('D_bare', 'or_at_least_one_v1', 'or_explicit_or_both_v1')


def read(path):
    return json.loads(Path(path).read_text())


class Inputs:
    def __init__(self):
        self.cfg = read(CONFIG)
        self.used = {}
        self.stats = {}
        for path, expected in self.cfg['input_pins'].items():
            self.verify(ROOT / path, expected)
        self.verify(Path(self.cfg['document_path']), self.cfg['document_pin'])
        self.bdelivery = read(ROOT / self.cfg['b25_delivery'])
        self.cdelivery = read(ROOT / self.cfg['constituent_delivery'])
        require(self.bdelivery['producer_sha'] == self.cfg['b25_producer'], 'B25 producer')
        require(self.cdelivery['producer_sha'] == self.cfg['constituent_producer'], 'constituent producer')
        self.broot = Path(self.bdelivery['external_results'])
        self.croot = Path(self.cdelivery['external_results'])
        bmp = Path(self.bdelivery['external_review']) / 'manifest.json'
        cmp = Path(self.cdelivery['external_manifest_path'])
        self.verify(bmp, self.bdelivery['external_manifest'])
        self.verify(cmp, self.cdelivery['external_manifest'])
        self.bmanifest = read(bmp)
        self.cmanifest = read(cmp)
        require(self.bmanifest['producer_sha'] == self.cfg['b25_producer'], 'B25 delivery manifest producer')
        require(self.cmanifest['producer_sha'] == self.cfg['constituent_producer'], 'constituent delivery manifest producer')
        self.bfiles = {str(Path(self.bmanifest['root']) / p): v for p, v in self.bmanifest['artifacts'].items()}
        self.cfiles = {str(Path(self.cdelivery['external_control']) / p): v for p, v in self.cmanifest['files'].items()}
        self.block = self.bread('locked.json')
        self.clock = self.cread('locked-selection.json')
        require(not self.block['wording_used_before_lock'] and not self.clock['D_or_wording_used'], 'selection exposure')
        for root, get in [(self.broot, self.bread), (self.croot, self.cread)]:
            completion = get('completion.json')
            require(completion['status'] == 'completed', 'completed fitting delivery required')
        acceptance = read(ROOT / 'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json')
        # Adapters reconstruct identities only. Their layer()/feature methods are never called.
        self.b = B25Adapter(acceptance)
        self.c = ConstituentAdapter(acceptance)
        self.acceptance = acceptance
        self.manifest = self.b.manifest
        self.fresh_cfg = self.b.cfg
        self.bweights_binding = self.bread('bootstrap_binding.json')
        self.cweights_binding = self.cread('bootstrap-binding.json')
        self.bverify('bootstrap_weights.npz', self.bweights_binding['weights'])
        self.cverify('bootstrap-weights.npy', self.cweights_binding['artifact'])
        with np.load(self.broot / 'bootstrap_weights.npz', allow_pickle=False) as d:
            self.weights = {g: d[g].copy() for g in OR_GROUPS + ('and_both_following_v1',)}
        cw = np.load(self.croot / 'bootstrap-weights.npy', allow_pickle=False)
        require(np.array_equal(cw, self.weights['D_bare']), 'paired panel schedule mismatch')
        require(self.bweights_binding['compound_schedule_sha256'] == self.cweights_binding['compound_schedule_sha256'], 'schedule identity')
        # Independent regeneration of the exact existing endpoint-product schedule.
        rng = np.random.Generator(np.random.PCG64(1729))
        rows = self.b.rows['D_bare']; rebuilt = np.zeros_like(cw)
        for topic in self.cfg['topics']:
            people = sorted({r[k] for r in rows if r['topic'] == topic for k in ('person_a', 'person_b')})
            lookup = {p: i for i, p in enumerate(people)}
            counts = rng.multinomial(len(people), np.full(len(people), 1 / len(people)), size=2000)
            idx = [i for i, r in enumerate(rows) if r['topic'] == topic]
            rebuilt[:, idx] = counts[:, [lookup[rows[i]['person_a']] for i in idx]] * counts[:, [lookup[rows[i]['person_b']] for i in idx]]
        require(np.array_equal(rebuilt, cw), 'bootstrap schedule regeneration')
        bare = {r['example_id']: i for i, r in enumerate(rows)}
        for g in OR_GROUPS[1:] + ('and_both_following_v1',):
            idx = [bare[r['example_id']] for r in self.b.rows[g]]
            require(np.array_equal(self.weights[g], cw[:, idx]), 'wording shared entity draws')
        self.bevals = {}
        for path in sorted((self.broot / 'evaluations').glob('*/score_binding.json')):
            # Only metadata is read to locate locked identities; unselected scores stay closed.
            binding = read(path)
            self.bevals[binding['fit_id']] = path.parent
        self.cpanels = {}
        for path in sorted((self.croot / 'scores').glob('*/binding.json')):
            binding = read(path)
            if binding['lock_sha256']:
                p = binding['packet']; head = binding['head_ids'][0]
                trained = head.split('/')[2]
                self.cpanels[p['model'], p['saved_layer'], trained, p['group'], p['operator']] = path.parent
        self.bfitpaths = {read(p)['id']: p for p in (self.broot / 'fits').glob('*/fit.json')}
        self.cfitpaths = {read(p)['id']: p for p in (self.croot / 'fits').glob('*/fit.json')}

    def verify(self, path, expected=None):
        path = Path(path).resolve(); key = str(path)
        if key not in self.used:
            self.used[key] = pin(path)
            self.stats[key] = (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        if expected is not None:
            require(self.used[key] == expected, 'hash/size mismatch: ' + key)
        return self.used[key]

    def bverify(self, rel, expected=None):
        path = self.broot / rel
        require(str(path) in self.bfiles, 'unmanifested B25 dependency: ' + str(path))
        value = self.verify(path, self.bfiles[str(path)])
        if expected is not None: require(value == expected, 'B25 receipt pin: ' + str(path))
        return value

    def cverify(self, rel, expected=None):
        path = self.croot / rel
        require(str(path) in self.cfiles, 'unmanifested constituent dependency')
        value = self.verify(path, self.cfiles[str(path)])
        if expected is not None: require(value == expected, 'constituent receipt pin')
        return value

    def bread(self, rel):
        self.bverify(rel); return read(self.broot / rel)

    def cread(self, rel):
        self.cverify(rel); return read(self.croot / rel)

    def b25(self, model, seed, arm):
        name = self.block['procedures'][model][str(seed)][arm]
        path = self.bfitpaths[name]; rel = str(path.relative_to(self.broot)); fit = self.bread(rel)
        require(fit['valid_for_scoring'] and fit['id'] == name and fit['model'] == model, 'locked fitting identity')
        folder = self.bevals[name]; rel = str(folder.relative_to(self.broot))
        binding = self.bread(rel + '/score_binding.json')
        require(binding['producer_sha'] == self.cfg['b25_producer'] and binding['fit_id'] == name, 'B25 scoring identity')
        require(binding['representation_manifest_sha256'] == self.acceptance['manifest_sha256'], 'B25 representation')
        require(binding['lock_sha256'] == file_hash(self.broot / 'locked.json'), 'B25 lock binding')
        self.bverify(rel + '/scores.npz', binding['artifact'])
        with np.load(folder / 'scores.npz', allow_pickle=False) as d:
            scores = {g: d[g].copy() for g in OR_GROUPS}
        for g, values in scores.items():
            p = binding['groups'][g]; bindings = self.b.bindings[g]
            require(p['row_binding_sha256'] == hash_value(bindings) and p['ordered_keys'] == [[r['group'], r['logical_id']] for r in bindings], 'B25 ordered scoring support')
            require(values.shape == (len(bindings),) and np.isfinite(values).all(), 'B25 saved score shape')
        metrics = self.bread(rel + '/metrics.json')
        self.bverify(rel + '/bootstrap_metrics.npy', metrics['draws'])
        return dict(model=model, seed=str(seed), arm=arm, head_id=name, saved_layer=fit['layer'],
                    score_folder=str(folder), scores=scores, metrics=metrics,
                    bootstrap=np.load(folder / 'bootstrap_metrics.npy', allow_pickle=False))

    def constituent(self, model, source, group, operator):
        layer = self.clock['selections'][model][source]['selected_layer']
        folder = self.cpanels[model, layer, source, group, operator]
        rel = str(folder.relative_to(self.croot)); binding = self.cread(rel + '/binding.json')
        require(binding['lock_sha256'] == file_hash(self.croot / 'locked-selection.json'), 'constituent lock')
        for h, position in zip(binding['head_ids'], ('FIRST', 'SECOND')):
            fit = self.cread(str(self.cfitpaths[h].relative_to(self.croot)))
            require(fit['valid'] and fit['saved_layer'] == layer and fit['model'] == model and fit['operator'] == source and fit['position'] == position and fit['group'] == 'TC_control_and_final_refit', 'selected constituent identity')
        self.cverify(rel + '/scores.npy', binding['artifact'])
        score = np.load(folder / 'scores.npy', allow_pickle=False)
        check_production_packet(self.c, binding['packet'], score)
        rows = self.c.block(group, operator)['rows']
        require(rows == [r for r in self.c.rows[group] if r['operator'] == operator], 'surface support')
        metrics = self.cread(rel + '/metrics.json')
        self.cverify(rel + '/bootstrap-metrics.npy', metrics['draws'])
        return dict(model=model, source_operator=source, evaluation_operator=operator, condition=group,
                    saved_layer=layer, head_ids=binding['head_ids'], score_folder=str(folder), rows=rows, scores=score,
                    metrics=metrics, bootstrap=np.load(folder / 'bootstrap-metrics.npy', allow_pickle=False))

    def unchanged(self):
        for key, before in self.stats.items():
            p = Path(key)
            require((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) == before, 'source changed: ' + key)
        self.b.unchanged(); self.c.unchanged()


def matched(bare_rows, word_rows):
    lookup = {r['example_id']: (i, r) for i, r in enumerate(bare_rows)}
    require(len(lookup) == len(bare_rows), 'duplicate bare identity')
    idx = []
    for r in word_rows:
        i, b = lookup[r['example_id']]
        for k in ('example_id', 'pair_id', 'person_a', 'person_b', 'fact_a_id', 'fact_b_id', 'truth_a', 'truth_b', 'operator', 'ordering', 'label', 'topic', 'split'):
            require(r[k] == b[k], 'matched wording identity changed: ' + k)
        idx.append(i)
    require(len(set(idx)) == len(word_rows), 'duplicate wording binding')
    return np.asarray(idx)
