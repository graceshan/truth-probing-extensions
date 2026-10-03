"""Manifest-resolved frozen references and eight raw comparators; no new metrics/fits."""
from pathlib import Path
import json
import numpy as np
from src.checkpoint_next_inventory import ROOT, inventory, digest, sha, require
from src.checkpoint_r2_fresh_inputs import hash_value
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_b25_store import Reusable, RECOVER_SHA, read
from src.checkpoint_r2_fresh_adapter import check_packet


class FrozenB25:
    def __init__(self,adapter):
        self.adapter=adapter
        self.delivery=read(ROOT/'results/checkpoint_r2_b25_convergence_fix_20261002/delivery.json')
        manifest_path=Path(self.delivery['external_review'])/'manifest.json'
        require(pin(manifest_path)==self.delivery['external_manifest'],'B25 delivery manifest')
        self.manifest=read(manifest_path);self.root=Path(self.delivery['external_results'])
        require(self.manifest['producer_sha']==self.delivery['producer_sha']=='5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4','completed B25 lineage')
        self.lock=self.json('locked.json');self.completion=self.json('completion.json')
        require(self.completion['status']=='completed' and not self.lock['wording_used_before_lock'],'frozen completed repair')
        self.evaluations={}
        for p in (self.root/'evaluations').glob('*/score_binding.json'):
            binding=read(p);self.evaluations[binding['fit_id']]=p.parent
        self.fits={read(p)['id']:p for p in (self.root/'fits').glob('*/fit.json')}

    def verify(self,path):
        path=Path(path);rel=str(path.relative_to(Path(self.manifest['root'])))
        require(rel in self.manifest['artifacts'] and pin(path)==self.manifest['artifacts'][rel],'manifest-bound B25 artifact: '+rel)
        return pin(path)

    def json(self,rel):
        path=self.root/rel;self.verify(path);return read(path)

    def head(self,model,seed,arm):
        name=self.lock['procedures'][model][str(seed)][arm]
        fitpath=self.fits[name];self.verify(fitpath);fit=read(fitpath)
        require(fit['valid_for_scoring'] and fit['model']==model,'frozen head valid/model')
        folder=self.evaluations[name];self.verify(folder/'score_binding.json');binding=read(folder/'score_binding.json')
        require(binding['producer_sha']==self.delivery['producer_sha'] and binding['fit_id']==name and
                binding['lock_sha256']==sha(self.root/'locked.json') and
                binding['representation_manifest_sha256']==self.adapter.manifest_hash,'frozen evaluation identity')
        self.verify(folder/'scores.npz');require(pin(folder/'scores.npz')==binding['artifact'],'frozen scores pin')
        return name,fit,folder,binding

    def describe(self,model,seed,arm):
        name,fit,folder,binding=self.head(model,seed,arm)
        return dict(head_id=name,model=model,seed=seed,arm=arm,saved_layer=fit['layer'],
            fit_path=str(self.fits[name]),fit_pin=pin(self.fits[name]),parameter_pin=fit['parameters'],
            parameter_producer_sha=fit.get('parameter_producer_sha',fit['producer_sha']),
            evaluation_binding_pin=pin(folder/'score_binding.json'),scores_path=str(folder/'scores.npz'),scores_pin=pin(folder/'scores.npz'),
            lock_sha256=sha(self.root/'locked.json'),representation_manifest_sha256=binding['representation_manifest_sha256'])

    def scores(self,model,seed,arm):
        name,fit,folder,binding=self.head(model,seed,arm)
        with np.load(folder/'scores.npz',allow_pickle=False) as z:
            result={}
            for g in ['atomic_D','D_bare']+sorted(self.adapter.receipt['heldout_wording_rows']):
                result[g]=check_packet(self.adapter.bindings[g],dict(binding['groups'][g],scores=z[g]))
        return result

    def existing_metrics(self,model,seed,arm):
        _,_,folder,_=self.head(model,seed,arm);self.verify(folder/'metrics.json')
        return read(folder/'metrics.json')['records']


def raw_exports(adapter):
    """Eight named heads, exact raw 7728 D scores and frozen 800-row chat support."""
    reusable=Reusable();x=inventory();groups={g:[b for b in x['bindings'] if b['role']=='chat' and b['group']==g]
        for g in ('P15','TC_control_and_final_refit','behavior_D')}
    export=[]
    for head in x['heads']:
        source=reusable.source(head['model'],head['layer'],head['family'])
        require(source is not None and source['valid_for_scoring'] and source['producer_sha']==RECOVER_SHA,'exact raw source')
        group='P15' if head['family']=='R0' else 'TC_control_and_final_refit'
        require(source['fit_rows']==len(adapter.rows[group]) and source['preprocessing_group']=='P15' and
            source['training_bindings_sha256']==hash_value(adapter.bindings[group]) and
            head['fitting_identities_sha256']==digest([b['logical_id'] for b in adapter.bindings[group]]),'raw fitting binding')
        require(head['preprocess_bindings_sha256']==digest(groups['P15']) and
                source['input_manifest_sha256']==adapter.manifest_hash,'raw preprocessing/representation')
        folder=reusable.root/'fits'/source['id']
        require(pin(folder/'fit.json')==reusable.completion['artifacts'][str((folder/'fit.json').relative_to(reusable.root))],'raw fit manifest pin')
        require(pin(folder/'head.npz')==source['parameters'] and pin(folder/'scores.npz')==source['scores'],'raw parameter/score pins')
        with np.load(folder/'scores.npz',allow_pickle=False) as z:
            scores=check_packet(adapter.bindings['D_bare'],dict(source['score_bindings']['D_bare'],scores=z['D_bare']))
        ids=[b['logical_id'] for b in adapter.bindings['D_bare']];support=[r['logical_id'] for r in groups['behavior_D']]
        require(len(ids)==7728 and len(support)==len(set(support))==800 and set(support)<=set(ids) and digest(support)==head['D_support_sha256'],'complete raw and matching chat support')
        lookup=dict(zip(ids,scores.tolist()))
        export.append(dict(head,id=head['id'],preprocessing_membership='original_P15',regime='raw',
            P15_identities_sha256=digest([b['logical_id'] for b in adapter.bindings['P15']]),
            logical_ids=ids,scores=scores.tolist(),support_800_logical_ids=support,support_800_scores=[lookup[i] for i in support],
            frozen_rule_source_sha256=digest(head),
            frozen_rule_provenance=dict(source='inherited frozen eight-head inventory',head_spec=head,
                behavior_config_sha256=sha(ROOT/'config/checkpoint_next/behavior_prep_v1.json'),
                inventory_implementation_sha256=sha(ROOT/'src/checkpoint_next_inventory.py'),
                decision='raw logit >=0; exact tie True; no D fit'),
            source_fit_path=str(folder/'fit.json'),source_fit_pin=pin(folder/'fit.json'),
            source_parameters_pin=source['parameters'],source_scores_pin=source['scores'],
            source_completion_pin=pin(reusable.root/'completion.json'),source_parameter_producer=RECOVER_SHA,
            original_fit_rows=source['fit_rows'],original_training_bindings_sha256=source['training_bindings_sha256'],
            original_feature_float64_sha256=source['feature_float64_sha256'],original_objective=source['objective'],
            raw_representation_manifest_sha256=source['input_manifest_sha256']))
    require(len(export)==8 and len({e['id'] for e in export})==8,'eight unique raw exports')
    reusable.unchanged()
    return export
