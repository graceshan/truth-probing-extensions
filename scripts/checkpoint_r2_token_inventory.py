"""Offline tokenizer-only inventory; no model weights or activations are opened."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import transformers
from transformers import AutoTokenizer


def digest(b):return hashlib.sha256(b).hexdigest()
def packed(x):return json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refuse overwrite')
    root=Path(__file__).resolve().parents[1]
    revision='a09a35458c702b33eeacc393d103063234e8bc28'
    if a.tokenizer.name!=revision:raise SystemExit('Wrong pinned Qwen tokenizer revision')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    source=root/'data/checkpoint_r2_v1/extraction_inventory.csv'
    rows=list(csv.DictReader(source.open()));groups=defaultdict(dict)
    for r in rows:groups[r['kind']][r['text_id']]=r['statement']
    output=[];summary={}
    for kind,texts in sorted(groups.items()):
        total=0;maximum=0;candidate_tokens=0;candidate_full=0
        for tid,text in sorted(texts.items()):
            if kind=='raw':ids=tok.encode(text,add_special_tokens=True)
            else:
                message=[{'role':'user','content':text}]
                rendered=tok.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
                ids=tok.apply_chat_template(message,tokenize=True,add_generation_prompt=True,return_dict=False)
                if ids!=tok.encode(rendered,add_special_tokens=False):raise ValueError('chat token boundary')
                if kind=='behavior':
                    for answer in ('True','False'):
                        full=tok.encode(rendered+answer,add_special_tokens=False)
                        if full[:len(ids)]!=ids:raise ValueError('answer changes prompt boundary')
                        candidate_tokens+=len(full)-len(ids);candidate_full+=len(full)
            total+=len(ids);maximum=max(maximum,len(ids))
            output.append(dict(kind=kind,text_id=tid,token_count=len(ids),token_ids_sha256=digest(packed(ids))))
        summary[kind]=dict(unique_inputs=len(texts),input_tokens=total,max_input_tokens=maximum)
        if kind=='behavior':summary[kind].update(candidate_content_tokens=candidate_tokens,candidate_scoring_tokens_without_prefix_reuse=candidate_full,greedy_new_token_cap=8*len(texts))
    bridge=json.loads((root/'data/checkpoint_r2_v1/bridge_identities.json').read_text())
    tokens={g:[dict(id=r['id'],token_ids=tok.encode(r['statement'],add_special_tokens=True),semantic_position=len(tok.encode(r['statement'],add_special_tokens=True))-1) for r in rs] for g,rs in bridge.items()}
    a.output.mkdir(parents=True)
    with (a.output/'token_bindings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n');w.writeheader();w.writerows(output)
    (a.output/'bridge_tokens.json').write_text(json.dumps(tokens,indent=2)+'\n')
    receipt=dict(model='Qwen/Qwen2.5-7B-Instruct',revision=revision,transformers_version=transformers.__version__,tokenizer_class=type(tok).__name__,native_template_sha256=digest(tok.chat_template.encode()),native_template_default_system='Native Qwen template inserts its own default system text for user-only input; no application-added system message or template edit',tokenizer_files={p.name:digest(p.read_bytes()) for p in a.tokenizer.iterdir() if p.name.startswith(('tokenizer','special_tokens','vocab','merges','chat_template'))},input_sha256=digest(source.read_bytes()),code_sha256=digest(Path(__file__).read_bytes()),counts=summary,answer_ids={s:tok.encode(s,add_special_tokens=False) for s in ('True','False')},native_eos_token_id=tok.eos_token_id,output_sha256={p.name:digest(p.read_bytes()) for p in a.output.iterdir()},status='offline_current_tokenization_only_not_historical_replay',llama='pending pinned tokenizer unavailable locally and remotely')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
