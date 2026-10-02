"""Bind frozen bridge IDs to restored historical metadata; never open activations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refuse overwrite')
    root=Path(__file__).resolve().parents[1];source=root/'data/checkpoint_r2_v1/bridge_identities.json';ids=json.loads(source.read_text())
    contracts={};bindings=[];bound_inputs={}
    paths={'qwen':{'train':'atomic_repair/qwen25_a09a354_bs1_bf16_v1/train','validation':'atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation','compound':'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1','probe':'atomic_probe_selection/qwen25_7b/selected_probe.npz'},'llama':{'train':'llama31_replication_v1/atomic/train','validation':'llama31_replication_v1/atomic/validation','compound':'llama31_replication_v1/transfer','probe':'llama31_replication_v1/probe/selected_probe.npz'}}
    for model,ps in paths.items():
        tables={}; manifests={}
        for kind in ('train','validation','compound'):
            path=a.payload_root/ps[kind]/'metadata.csv';raw=list(csv.DictReader(path.open()));bound_inputs[str(path)]=digest(path)
            indexed={}
            for i,r in enumerate(raw):
                if 'condition_id' in r and r['condition_id']!='raw_reference':continue
                key=r.get('example_id') or r['dataset']+':'+r['row_index']
                if key in indexed:raise ValueError('duplicate metadata identity')
                indexed[key]=(i,r)
            tables[kind]=indexed
            manifest=a.payload_root/ps[kind]/'extraction_manifest.json'
            if not manifest.exists():manifest=manifest.parent.parent/'extraction_manifest.json'
            bound_inputs[str(manifest)]=digest(manifest);m=json.loads(manifest.read_text())
            manifests[kind]={k:m[k] for k in ('contract','pin','runtime','resolved_model','representation_fingerprint','representation_descriptor','execution_contract','numerics','representation') if k in m}
        for stage,rs in ids.items():
            for r in rs:
                kind='compound' if r['kind']=='compound' else r['split'];i,old=tables[kind][r['id']]
                if r['statement']!=old['statement'] or r['split']!=old['split']:raise ValueError('historical binding mismatch')
                bindings.append(dict(model=model,stage=stage,id=r['id'],split=r['split'],statement_sha256=hashlib.sha256(r['statement'].encode()).hexdigest(),historical_index=i,metadata_path=ps[kind]+'/metadata.csv',remote_tensor='/workspace/truth-probing-artifacts/'+ps[kind]+'/activations.npy'))
        probe=a.payload_root/ps['probe'];bound_inputs[str(probe)]=digest(probe)
        contracts[model]=dict(recorded_manifests=manifests,frozen_probe=dict(path=str(probe),sha256=digest(probe),saved_layer=17 if model=='qwen' else 15,C=1 if model=='qwen' else 10,role='historical frozen canonical head for compatibility diagnostics only; no accuracy/AUROC or new fit'),historical_token_ids='not available in restored sidecars',execution_verification='recorded settings only; execution identity/provenance conflicts remain unresolved')
    out=dict(bridge_source_sha256=digest(source),code_sha256=digest(Path(__file__)),bound_inputs=bound_inputs,bindings=bindings,contracts=contracts,status='metadata_bindings_passed_not_representation_compatibility',activation_values_read=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('Bound',len(bindings),'model/stage/row references')

if __name__=='__main__':main()
