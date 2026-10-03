"""Future verified 50-pair behavior report with optional frozen full-D probe exports."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_next_runner import frozen,verify,producer
from src.checkpoint_next_reporting import report,expand,probe_behavior_comparison,verified_chat_heads,chat_behavior_comparison
from src.checkpoint_next_store import publish_json
from src.checkpoint_next_inventory import require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);p.add_argument('--model',required=True,choices=['qwen','llama']);p.add_argument('--probes');p.add_argument('--chat-fits');p.add_argument('--raw-matches')
    a=p.parse_args();verified=verify(a.input,'launch',a.model);x=frozen();records=[]
    for path in sorted((Path(a.input)/'launch'/a.model/'shards').glob('*/records.json')):records.extend(json.loads(path.read_text()))
    result=dict(verification=verified,behavior=[report(x,records,p) for p in (1,2)])
    if a.probes:
        probes=json.loads(Path(a.probes).read_text())
        require(all(p['model']==a.model for p in probes),'probe model mismatch')
        result['probe_comparison']=[dict(prompt=p,**probe_behavior_comparison(x,[r for r in expand(x,records,p) if r['group']=='behavior_compound'],probe)) for p in (1,2) for probe in probes]
    else:result['probe_comparison_status']='pending verified frozen full-D exports; behavior-only report cannot complete checkpoint'
    if a.chat_fits:
        require(a.raw_matches,'eight reviewed full-D raw exports required for chat adapter')
        raw=json.loads(Path(a.raw_matches).read_text());heads=verified_chat_heads(x,a.chat_fits,producer(),raw)
        rows=[r for r in expand(x,records,1) if r['group']=='behavior_compound']
        lookup={r['id']:r for r in raw}
        result['matched_chat']=[chat_behavior_comparison(rows,h,lookup[h['id']]) for h in heads if h['model']==a.model]
        result['full_D_raw_separate']=[probe_behavior_comparison(x,rows,dict(r,head_id=r['id'])) for r in raw if r['model']==a.model]
    result['model']=a.model
    publish_json(a.output,result)


if __name__=='__main__':main()
