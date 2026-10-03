"""Validated production constituent bindings; synthetic common API stays unchanged."""
import csv
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoint_r2_fresh_adapter import FreshAdapter
from src.checkpoint_r2_b25_adapter import resolve
from src.checkpoint_r2_common_v1 import FrozenBindings
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require, text_hash
from src.checkpoint_r2_full_inputs import WORDING
from src.checkpoint_r2_selection_v1 import TOPICS, surface_labels, select_constituent_layer

FIT='constituent_source_fit'
VALID='constituent_source_validation'
FULL='TC_control_and_final_refit'
BARE='D_bare'
TRAIN=(FIT,FULL)


def exclusion_gate(rows, p_rows, entities):
    """Verify frozen 4/16 per-topic roles, A15 validation and P15 exclusion."""
    people=lambda group:{r[k] for r in rows[group] for k in ('person_a','person_b')}
    fit, valid, full = (people(g) for g in (FIT,VALID,FULL))
    p={r['person_key'] for r in p_rows}
    a={r['person_key'] for r in entities if r['A15']}
    require(len(valid)==20 and valid<=a and not valid&(fit|p), 'source-validation person exclusion')
    require(len(fit)==80 and len(full)==100 and fit|valid==full and not p&a, 'frozen TC/P15 person contract')
    for t in TOPICS:
        sets=[{r[k] for r in rows[g] if r['topic']==t for k in ('person_a','person_b')} for g in (FIT,VALID,FULL)]
        require(list(map(len,sets))==[16,4,20], 'topic 16/4/20 person counts')
        for g,n in ((FIT,100),(VALID,6),(FULL,100)):
            rr=[r for r in rows[g] if r['topic']==t]
            require(len({r['pair_id'] for r in rr})==n and len(rr)==n*16,'frozen topic pair coverage')
        vv={tuple(sorted((r['person_a'],r['person_b']))) for r in rows[VALID] if r['topic']==t}
        require(vv==set(combinations(sorted(sets[1]),2)), 'all six validation pairs')
    return dict(validation_people=sorted(valid),fitting_people=sorted(fit),full_TC_people=sorted(full),
                validation_excluded_from_P15_and_source_fit=True,validation_A15=True)


class ConstituentAdapter(FreshAdapter):
    def __init__(self, acceptance):
        super().__init__(acceptance['output'],acceptance)
        self.frozen=FrozenBindings(ROOT)
        compounds=list(csv.DictReader((ROOT/'data/checkpoint_r2_v1/compound_bindings.csv').open()))
        for g in (FIT,VALID):
            self.sources[g]=[r for r in compounds if r['group']==g]
            self.rows[g]=[self.frozen.adapt_row(r) for r in self.sources[g]]
            self.bindings[g]=resolve(g,self.sources[g],self.manifest,'example_id')
        from src.checkpoint_r2_inputs import uid
        from src.final_e_preparation import render_wording
        from src.heldout_and_wording import render as render_and
        lookup={(b['group'],b['logical_id']):b for b in self.manifest['bindings']}
        texts={r['id']:i for i,r in enumerate(self.manifest['rows'])}
        bare=[r for r in compounds if r['group']==BARE]
        for g in sorted(WORDING):
            self.sources[g]=[];self.rows[g]=[];self.bindings[g]=[]
            for raw in bare:
                if g.startswith('and_')!=(raw['operator']=='AND'):continue
                adapted=self.frozen.adapt_row(raw)
                sides=('a','b') if raw['ordering']=='AB' else ('b','a')
                a,b=[self.frozen.facts[raw['fact_'+s+'_id']]['statement'] for s in sides]
                statement=render_and(a,b,'AND') if raw['operator']=='AND' else render_wording(g,a,b)
                source=dict(raw,group=g,statement=statement)
                logical=uid('r2wording',raw['example_id'],g);binding=lookup[g,logical]
                require(binding['source_metadata_sha256']==hash_value(source),'wording source metadata')
                self.sources[g].append(source)
                self.rows[g].append(dict(adapted,group=g,statement=statement,role='D_heldout_wording',logical_id=logical))
                self.bindings[g].append(dict(group=g,logical_id=logical,text_id=binding['text_id'],text_row=texts[binding['text_id']],
                    inventory_offset=binding['inventory_offset'],source_metadata_sha256=binding['source_metadata_sha256']))
            require(len(self.rows[g])==3864,'wording coverage')
        # Validate every explicit source and observation, not just total counts.
        for g in (FIT,VALID,FULL,BARE,*sorted(WORDING)):
            require(len(self.rows[g])==len(self.bindings[g]),'logical coverage')
            for source,b in zip(self.sources[g],self.bindings[g]):
                binding=lookup[g,b['logical_id']]
                require(binding in self.manifest['rows'][b['text_row']]['bindings'],'binding/text row membership')
                require(binding['text_id']=='text_'+text_hash(source['statement']),'exact text ID/hash')
                require(binding['statement']==source['statement'] and binding['label']==str(source['label']),'statement/label binding')
                require(binding['split']==('train' if g in (FIT,VALID,FULL) else 'validation'),'role split')
                require(binding['source_metadata_sha256']==hash_value(source),'source/person/fact identity hash')
            offsets=[b['inventory_offset'] for b in self.bindings[g]]
            require(offsets==sorted(offsets) and len({b['logical_id'] for b in self.bindings[g]})==len(offsets),'logical order/duplicates')
            if g not in WORDING:
                for op in ('AND','OR'):
                    rr=[r for r in self.rows[g] if r['operator']==op]
                    counts=Counter(r['pair_id'] for r in rr)
                    require(all(n==8 for n in counts.values()),'complete operator/cell/order exposure')
                    for pair in counts:
                        require({(r['truth_a'],r['truth_b'],r['ordering']) for r in rr if r['pair_id']==pair}==set(product((0,1),(0,1),('AB','BA'))),'truth/order coverage')
            surface_labels(pd.DataFrame(self.rows[g]))
        entities=json.loads((ROOT/'data/checkpoint_r2_v1/entities.json').read_text())
        self.exclusions=exclusion_gate(self.rows,self.rows['P15'],entities)
        # Only constituent groups and P15 are exposed to this production runner.
        allowed=('P15',FIT,VALID,FULL,BARE,*sorted(WORDING))
        for attr in ('rows','sources','bindings'):
            value=getattr(self,attr);setattr(self,attr,{g:value[g] for g in allowed})
        self.receipt.update(schema='r2-fresh-constituent-production-bindings-v1',exclusions=self.exclusions,
            groups={g:dict(rows=len(self.rows[g]),logical_bindings_sha256=hash_value(self.bindings[g]),
                           source_metadata_sha256=hash_value(self.sources[g]),physical_texts=len({b['text_row'] for b in self.bindings[g]})) for g in allowed},
            score_contract='fresh_production_constituent_v1; representation+ordered logical metadata+artifact hashes',
            synthetic_API_unchanged=True)

    def indices_for(self,group,operator):
        require(group in self.rows and operator in ('AND','OR'),'declared constituent block')
        return [i for i,r in enumerate(self.rows[group]) if r['operator']==operator]

    def block(self,group,operator):
        idx=self.indices_for(group,operator)
        return dict(group=group,operator=operator,rows=[self.rows[group][i] for i in idx],bindings=[self.bindings[group][i] for i in idx])

    def layer_groups(self,model,layer,groups):
        require(set(groups)<=set(self.bindings),'undeclared/E group')
        original=self.bindings
        try:
            self.bindings={g:original[g] for g in groups}
            return super().layer(model,layer)
        finally:self.bindings=original

    def select_production(self,model,operator,scores_by_layer):
        block=self.block(VALID,operator)
        require(set(scores_by_layer)==set(range(self.cfg['models'][model]['layers'])),'complete source layer inventory')
        return select_constituent_layer(source_operator=operator,role='T_C_source_validation',
            validation_binding=hash_value(block['bindings']),rows=pd.DataFrame(block['rows']),scores_by_layer=scores_by_layer)


def score_packet(adapter,model,layer,group,operator,values):
    block=adapter.block(group,operator)
    packet=dict(score_kind='fresh_production_constituent_v1',model=model,saved_layer=layer,group=group,operator=operator,
        representation_index_sha256=file_hash(adapter.output/model/'index.json'),
        row_binding_sha256=hash_value(block['bindings']),ordered_keys=[[b['group'],b['logical_id']] for b in block['bindings']])
    check_production_packet(adapter,packet,values)
    return packet


def check_production_packet(adapter,packet,values):
    require(packet['score_kind']=='fresh_production_constituent_v1','production contract required')
    block=adapter.block(packet['group'],packet['operator'])
    require(packet['representation_index_sha256']==file_hash(adapter.output/packet['model']/'index.json'),'representation packet mismatch')
    require(packet['row_binding_sha256']==hash_value(block['bindings']) and packet['ordered_keys']==[[b['group'],b['logical_id']] for b in block['bindings']],'ordered production scores mismatch')
    require(type(packet['saved_layer']) is int and 0<=packet['saved_layer']<adapter.cfg['models'][packet['model']]['layers'],'layer identity')
    require(np.asarray(values).shape==(len(block['rows']),2) and np.isfinite(values).all(),'production score dimensions/finite')
    return np.asarray(values)
