"""Manifest-resolved frozen heads. Only selected saved parameters; no fitting/search."""
from pathlib import Path
import numpy as np
from src.final_e_inventory import ROOT, BASE, read
from src.checkpoint_r2_fresh_inputs import hash_value, require, file_hash
from src.checkpoint_r2_fresh_store import pin
from src.selection_repair_capacity_methods_v1 import score


class Source:
    def __init__(self, name, root, manifest, expected, prefix=''):
        self.name, self.root = name, Path(root)
        require(pin(manifest) == expected, 'reviewed manifest pin: '+name)
        self.manifest_pin = expected
        d = read(manifest)
        self.files = d.get('artifacts', d.get('files'))
        self.prefix = prefix

    def ref(self, relative):
        p = self.root/relative
        expected = self.files[self.prefix+relative]
        require(pin(p) == expected, 'reviewed artifact changed: '+self.name+'/'+relative)
        return dict(source=self.name, path=relative, **expected)

    def json(self, relative):
        self.ref(relative)
        return read(self.root/relative)


def build(roots):
    """Roots are relocation hints; trusted manifests/pins determine contents."""
    b = read(ROOT/'results/checkpoint_r2_b25_convergence_fix_20261002/delivery.json')
    c = read(ROOT/'results/checkpoint_r2_fresh_constituents_20261003/delivery.json')
    bp, cp, fp = map(Path, (roots['b25'], roots['constituent'], roots['full_bank']))
    bm = Path(roots['b25_manifest']); cm = Path(roots['constituent_manifest'])
    fm = ROOT/'results/checkpoint_next_full_bank_results_20261003/external-output-manifest.json'
    require(file_hash(cm) == 'e082f09a481196014a48c53f659d48b6ec0339e6990402cde6fdf1d5c1238f27', 'constituent delivery manifest')
    sources = dict(b25=Source('b25', bp, bm, b['external_manifest'], bp.name+'/'),
                   constituent=Source('constituent', cp, cm, pin(cm), cp.name+'/'),
                   full_bank=Source('full_bank', fp, fm, pin(fm), 'bank/'))
    heads, aliases, cases, evidence = {}, [], [], {}
    def add(src, name, family):
        key = src.name+':'+name
        if key in heads:
            return key
        folder = 'fits/'+hash_value(name)[:24]
        fit = src.json(folder+'/fit.json')
        require(fit.get('valid_for_scoring', fit.get('valid')) is True, 'invalid frozen head')
        if family == 'constituent':
            model, layer, method = fit['model'], fit['saved_layer'], 'constituent'
            parameter = folder+'/head.npz'
            prep = f'preprocessing/{model}/L{layer:02d}.npz'
            pre = src.ref(prep)
            require({k:pre[k] for k in ('bytes','sha256')} == fit['preprocessing_artifact'], 'constituent preprocessing pin')
            premeta = src.ref(prep.replace('.npz','.json'))
            producer = fit['producer_sha']
        else:
            spec = fit.get('spec', {})
            model, layer = fit.get('model', spec.get('model')), fit.get('layer', spec.get('saved_layer'))
            method = fit.get('scoring_method', spec.get('method'))
            parameter, pre, premeta = folder+'/parameters.npz', None, None
            producer = fit.get('parameter_producer_sha', fit.get('params_producer', fit.get('producer_sha')))
        par = src.ref(parameter)
        require({k:par[k] for k in ('bytes','sha256')} == fit['parameters'], 'parameter pin')
        require(type(layer) is int and 0 <= layer < (28 if model == 'qwen' else 32), 'zero-based layer')
        require(isinstance(producer,str) and len(producer)==40, 'parameter producer')
        heads[key] = dict(id=key, frozen_id=name, family=family, model=model, saved_layer=layer, method=method,
            parameter_producer=producer, fit=src.ref(folder+'/fit.json'), parameters=par,
            preprocessing=pre, preprocessing_metadata=premeta,
            preprocessing_policy='saved P15 standardizer' if pre else 'exact saved parameter archive; method-specific preprocessing included',
            decision_rule='logit >= 0; ties True' if family=='constituent' else 'AUROC only; no invented decision threshold',
            status='verified_saved_valid_head', representation='fresh raw last real input BF16/FP16; zero-based hidden_states[1:]')
        return key
    bs=sources['b25']; lock=bs.json('locked.json')
    evidence['b25_lock']=bs.ref('locked.json')
    for model, seeds in lock['procedures'].items():
        require(set(seeds)=={'11','23','37'}, 'all B25 samples')
        for seed, arms in seeds.items():
            for arm, name in arms.items():
                category = 'descriptive_D_selected_reference' if arm in ('bank_oracle','C_clean_D_selected') else 'sensitivity' if 'sensitivity' in arm else 'original_B25_comparison_or_control'
                aliases.append(dict(panel='b25',model=model,sample=int(seed),arm=arm,head=add(bs,name,'scalar'),category=category))
    fs=sources['full_bank']; lock=fs.json('locked.json'); evidence['full_bank_lock']=fs.ref('locked.json')
    cfg=read(ROOT/'config/checkpoint_next/full_bank_prep_v1.json')
    from src.checkpoint_r2_selection_v1 import candidate_id
    for model, ml in lock['models'].items():
        for seed in (11,23,37):
            names=dict(ml['procedures'][str(seed)],full_bank_eligible=ml['oracle']['eligible']['fit_id'],
                       full_bank_unrestricted=ml['oracle']['unrestricted']['fit_id'])
            fixed=cfg['models'][model]
            names['full_canonical_LR']=candidate_id(model,fixed['fixed_layer'],'l2_logistic',fixed['fixed_C']).replace('/P15/','/full_train/')
            for arm,name in names.items():
                key=next(a['head'] for a in aliases if a['panel']=='b25' and a['model']==model and a['sample']==seed and a['arm']=='reduced_LR') if name=='frozen:reduced_LR' else add(fs,name,'scalar')
                aliases.append(dict(panel='full_bank',model=model,sample=seed,arm=arm,head=key,
                    category='descriptive_D_selected_reference' if arm.startswith('full_bank_') else 'full_training_sensitivity'))
    cs=sources['constituent']; inventory=cs.json('final-fitting-inventory.json')
    evidence['constituent_lock']=cs.ref('locked-selection.json');evidence['constituent_inventory']=cs.ref('final-fitting-inventory.json')
    for a in inventory['alias_records']:
        model, source=a['model'],a['source_operator'];target='OR' if source=='AND' else 'AND'
        for arm, trained, op in [('source_on_source',source,source),('source_on_target',source,target),('target_on_target',target,target)]:
            selected=[add(cs,n,'constituent') for n in a['heads'][trained]]
            require([cs.json(heads[k]['fit']['path'])['position'] for k in selected]==['FIRST','SECOND'], 'surface head order')
            cases.append(dict(model=model,source_operator=source,target_operator=target,arm=arm,trained_operator=trained,
                              evaluation_operator=op,heads=selected,saved_layer=heads[selected[0]]['saved_layer']))
    return dict(schema='final-e-frozen-head-registry-v1',base=BASE,status='DRAFT_proposed_for_E_not_lock',
        source_manifests={k:s.manifest_pin for k,s in sources.items()},selection_evidence=evidence,
        heads=heads,scalar_aliases=aliases,constituent_cases=cases,
        counts=dict(unique_heads=len(heads),scalar_aliases=len(aliases),constituent_heads=sum(h['family']=='constituent' for h in heads.values()),constituent_cases=len(cases)),
        primary_proposal=['B25 R_all versus S_all','S_all versus S_atom_all','frozen constituent cross-operator transfer'],
        primary_approval='pending final scientific lock',E_selection=False,all_600_E_search=False)


def resolve(ref, roots):
    root=Path(roots[ref['source']]).resolve();p=root/ref['path']
    require(p.resolve().is_relative_to(root) and pin(p)=={k:ref[k] for k in ('sha256','bytes')}, 'frozen artifact hash/path')
    return p


def apply_head(head, X, roots):
    fit=read(resolve(head['fit'],roots))
    require(fit.get('valid_for_scoring',fit.get('valid')) is True, 'failed head cannot score')
    spec=fit.get('spec', {})
    require(head['frozen_id']==fit.get('id',spec.get('fit_id')) and head['model']==fit.get('model',spec.get('model'))
            and head['saved_layer']==fit.get('layer',fit.get('saved_layer',spec.get('saved_layer'))), 'altered head identity/layer')
    require(head['parameter_producer']==fit.get('parameter_producer_sha',fit.get('params_producer',fit.get('producer_sha'))), 'altered parameter producer')
    expected_method='constituent' if head['family']=='constituent' else fit.get('scoring_method',spec.get('method'))
    require(head['method']==expected_method, 'altered method/preprocessing')
    require({k:head['parameters'][k] for k in ('sha256','bytes')}==fit['parameters'], 'altered parameter reference')
    path=resolve(head['parameters'],roots)
    with np.load(path,allow_pickle=False) as z:params={k:z[k] for k in z.files}
    X=np.asarray(X,dtype=np.float64)
    require(X.ndim==2 and X.shape[1]==(3584 if head['model']=='qwen' else 4096) and np.isfinite(X).all(), 'feature width/nonfinite')
    if head['family']=='constituent':
        resolve(head['preprocessing_metadata'],roots)
        with np.load(resolve(head['preprocessing'],roots),allow_pickle=False) as p:
            require({k:head['preprocessing'][k] for k in ('sha256','bytes')}==fit['preprocessing_artifact'], 'altered preprocessing reference')
            Z=np.zeros_like(X);active=~p['constant_mask'].astype(bool)
            Z[:,active]=(X[:,active]-p['mean'][active])/p['denominator'][active]
        value=Z@params['weights']+float(params['intercept'])
    else:
        value=score(head['method'],params,X)
    require(value.shape==(len(X),) and np.isfinite(value).all(), 'frozen score shape/nonfinite')
    return value
