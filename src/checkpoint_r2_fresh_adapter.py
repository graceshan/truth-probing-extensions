"""Identity-bound fresh raw feature adapter; no historical parameters or arrays."""
import csv
import json
from pathlib import Path

import numpy as np

from src.checkpoint_r2_common_v1 import FrozenBindings, membership_receipt
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require
from src.checkpoint_r2_full_raw import plan
from src import selection_repair_objectives as objective

GROUPS = ('P15', 'atomic_D', 'TC_control_and_final_refit', 'D_bare')
TRAIN = {'P15', 'TC_control_and_final_refit'}


def bind_rows(group, sources, manifest):
    """Lookup by logical identity; return original observations, never deduplicate."""
    require(group in GROUPS, 'undeclared feature group')
    lookup = {(b['group'], b['logical_id']): b for b in manifest['bindings']}
    require(len(lookup) == len(manifest['bindings']), 'duplicate logical binding')
    text_lookup = {r['id']: i for i, r in enumerate(manifest['rows'])}
    require(len(text_lookup) == len(manifest['rows']), 'duplicate extracted text ID')
    records = []
    for source in sources:
        logical = source['source_row_id'] if group in ('P15', 'atomic_D') else source['example_id']
        key = (group, logical)
        require(key in lookup, 'missing logical binding: ' + str(key))
        b = lookup[key]
        require(b['source_metadata_sha256'] == hash_value(source), 'source metadata binding')
        require(b['statement'] == source['statement'] and b['label'] == str(source['label']), 'source statement/label binding')
        require(b['split'] == ('train' if group in TRAIN else 'validation'), 'training/evaluation split')
        index = text_lookup[b['text_id']]
        require(b in manifest['rows'][index]['bindings'], 'binding assigned to wrong text row')
        require(b['text_id'] == 'text_' + __import__('hashlib').sha256(source['statement'].encode()).hexdigest(), 'source text hash')
        records.append(dict(group=group, logical_id=logical, text_id=b['text_id'], text_row=index,
                            inventory_offset=b['inventory_offset'], source_metadata_sha256=b['source_metadata_sha256']))
    require(len({(r['group'],r['logical_id']) for r in records}) == len(records), 'duplicate logical observation')
    require([r['inventory_offset'] for r in records] == sorted(r['inventory_offset'] for r in records), 'frozen source order mismatch')
    return records


def check_packet(records, packet):
    require(packet['row_binding_sha256'] == hash_value(records), 'stale row binding hash')
    require(packet['ordered_keys'] == [[r['group'],r['logical_id']] for r in records], 'score order/binding mismatch')
    values = np.asarray(packet['scores'], dtype=np.float64)
    require(values.shape == (len(records),) and np.isfinite(values).all(), 'score shape/nonfinite')
    return values


def require_fit_group(group):
    require(group in TRAIN, 'evaluation/wording group forbidden in fitting')


def clean_block(features, rows, pre):
    """Mean TC BCE adapter for the reviewed mean-block solver, with P statistics."""
    require(all(r['group'] == 'TC_control_and_final_refit' and r['split'] == 'train' for r in rows), 'C_clean requires frozen full TC training block')
    weights = objective.compound_pair_weights(
        [r['pair_id'] for r in rows], [r['operator'] for r in rows],
        [(int(r['truth_a']),int(r['truth_b'])) for r in rows],
        [0 if r['ordering']=='AB' else 1 for r in rows])
    require(np.allclose(weights, 1/len(rows), rtol=0, atol=1e-15), 'C_clean requires complete equal-exposure pairs')
    # fit_readout('r0', block, pre) is the reviewed mean-single-block numerical
    # primitive. Here the block is TC, not P: no half P / half TC repair loss.
    return objective.prepare_block(features, [int(r['label']) for r in rows], pre, weights)


class FreshAdapter:
    def __init__(self, output, acceptance, root=ROOT):
        self.output = Path(output).resolve()
        require(acceptance['status'] == 'independently_accepted_full_fresh_raw', 'input acceptance required')
        require(self.output == Path(acceptance['output']).resolve(), 'acceptance output path mismatch')
        self.cfg, self.manifest = plan()
        require(hash_value(self.manifest) == acceptance['manifest_sha256'], 'accepted representation manifest changed')
        self.membership = membership_receipt(root, rebuild=False)
        base = root / 'data/checkpoint_r2_v1'
        members = json.loads((base / 'memberships.json').read_text())
        entities = json.loads((base / 'entities.json').read_text())
        frozen = FrozenBindings(root)
        with (base / 'compound_bindings.csv').open() as f:
            compound = list(csv.DictReader(f))
        self.rows = {'P15': members['P15'], 'atomic_D': members['atomic_D']}
        self.sources = dict(self.rows)
        for group in GROUPS[2:]:
            self.sources[group] = [r for r in compound if r['group'] == group]
            self.rows[group] = [frozen.adapt_row(r) for r in self.sources[group]]
        self.bindings = {g: bind_rows(g,self.sources[g],self.manifest) for g in GROUPS}
        require([len(self.rows[g]) for g in GROUPS] == [2778,1012,8000,7728], 'early-stage row counts')
        a = {r['person_key'] for r in entities if r['A15']}
        p = {r['person_key'] for r in self.rows['P15']}
        tc = {r['person_key'] for r in entities}
        d = {r['person_key'] for r in self.rows['atomic_D']}
        dc = {r[k] for r in self.rows['D_bare'] for k in ('person_a','person_b')}
        require(len(a)==75 and len(tc)==100 and not a&p, 'A-person/source-variant exclusion')
        require(not (p|tc)&(d|dc), 'training/development person leakage')
        require(len({r['pair_id'] for r in self.rows['TC_control_and_final_refit']})==500, 'TC pairs')
        require(len({r['pair_id'] for r in self.rows['D_bare']})==483, 'D pairs')
        self.receipt = dict(memberships=self.membership, group_rows={g:len(self.rows[g]) for g in GROUPS},
                            ordered_row_bindings={g:hash_value(b) for g,b in self.bindings.items()},
                            source_rows={g:hash_value(r) for g,r in self.sources.items()},
                            distinct_text_rows={g:len({r['text_row'] for r in b}) for g,b in self.bindings.items()},
                            all_A_variants_excluded=True, TC_D_person_separation=True,
                            fresh_representation_only=True, historical_matches_admitted=0)
        self.indices = {}
        for model in self.cfg['models']:
            path = self.output/model/'index.json'
            require(file_hash(path)==acceptance['models'][model]['index_sha256'], 'accepted model index changed')
            self.indices[model] = json.loads(path.read_text())
        self.file_stats = {p: (p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
                           for p in self.output.rglob('*') if p.is_file()}

    def layer(self, model, layer):
        """One saved layer, mapped back to all logical observations in source order."""
        spec = self.cfg['models'][model]
        require(type(layer) is int and 0<=layer<spec['layers'], 'saved layer out of range')
        result = {g:np.empty((len(b),spec['width']),dtype=np.float64) for g,b in self.bindings.items()}
        filled = {g:np.zeros(len(b),bool) for g,b in self.bindings.items()}
        start = 0
        for shard in self.indices[model]['shards']:
            path = self.output/model/'shards'/shard['path']/'activations.npy'
            require((path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)==self.file_stats[path], 'accepted features changed during fitting')
            values = np.load(path,mmap_mode='r',allow_pickle=False)
            require(values.shape==(shard['rows'],spec['layers'],spec['width']) and values.dtype==np.float16, 'feature layer axes/dtype')
            stop = start + shard['rows']
            for g, bindings in self.bindings.items():
                dst = [i for i,r in enumerate(bindings) if start<=r['text_row']<stop]
                if dst:
                    src = [bindings[i]['text_row']-start for i in dst]
                    result[g][dst] = values[src,layer,:]
                    filled[g][dst] = True
            del values
            start = stop
        require(start==len(self.manifest['rows']) and all(v.all() for v in filled.values()), 'missing feature rows')
        require(all(np.isfinite(v).all() for v in result.values()), 'nonfinite fresh features')
        return result

    def unchanged(self):
        require(all((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)==s for p,s in self.file_stats.items()), 'read-only backup changed')
