#!/usr/bin/env python3
"""Render the audited post-hoc delivery; no E scoring or bootstrap calculations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in records])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/posthoc_reduced_bank_e_20261011')
    args=p.parse_args();out=args.results
    execution=read(out/'execution.json');audit=read(out/'audit.json')
    if audit['status']!='PASS':raise ValueError('A passing independent audit is required')
    summary=csvrows(out/'summary.csv');winners=csvrows(out/'winning_candidates.csv');fixed=csvrows(out/'fixed_procedures.csv')
    def fmt(x):return f'{float(x):.6f}'
    columns=['model','scope','S_all','maximum','R_all','maximum_minus_S_all','R_all_minus_maximum']
    rows=[[r[k] if k in ('model','scope') else fmt(r[k]) for k in columns] for r in summary]
    winner_rows=[[r['model'],r['maximum_scope'],r['candidate_id'],r['method'],r['saved_layer'],r['C'] or 'none',r['exact_maximum']] for r in winners]
    allocation_rows=[]
    for model in ('qwen','llama'):
        for arm in ('S_all','R_all'):
            rr=[r for r in fixed if r['model']==model and r['arm']==arm]
            allocation_rows.append([model,arm,*[fmt(next(r['E_auroc'] for r in rr if r['sample']==s)) for s in ('11','23','37','mean_11_23_37')]])
    q=next(r for r in summary if r['model']=='qwen' and r['scope']=='unrestricted')
    l=next(r for r in summary if r['model']=='llama' and r['scope']=='unrestricted')
    # Authored interpretation here is appropriate only after checking this outcome.
    assert all(float(r['R_all_minus_maximum'])>0 for r in summary)
    assert all(next(r['maximum'] for r in summary if r['model']==model and r['scope']=='eligible')==
               next(r['maximum'] for r in summary if r['model']==model and r['scope']=='unrestricted') for model in ('qwen','llama'))
    runtime=execution['runtime']
    report=f'''# Post-hoc reduced-bank maximum on E — executed and audited

**PASS.** All 600 frozen original reduced/P15 candidates were scored: 280 Qwen and 320 Llama. Original atomic-D eligibility retains 144 and 182, respectively. Eligible and unrestricted maxima coincide. No tested candidate matches or exceeds the unchanged joint-training mean on E.

This is explicitly authorized **post-hoc** analysis. E labels identify the winner. The maximum is an optimistic descriptive reference on this observed test set, not an independently evaluated selection procedure or a bound on future-population performance. Its scope is the finite tested bank, not every possible atomic-trained probe. The original selected heads, primary comparisons, scientific lock and manuscript are unchanged. Compound D historically informed exploratory development; a separate October 10 analysis already examined E. This specification was saved before this execution and makes no prospective or first-use claim.

## Results

Endpoint: bare pooled OR TF/FT versus FF, excluding TT, using the accepted E identities and score directions. There are 450 pairs, 2,700 boundary rows (1,800 positive and 900 negative), and both surface orders. Original atomic-D eligibility is unchanged; atomic E is not used. Each score tie receives half credit.

{table(['Model','Maximum scope','S_all','E maximum','R_all','Maximum − S_all','R_all − maximum'],rows)}

Both maxima use the same unique winner in each model. Exact maximizers and maximum-relative 1e-12 tolerance ties were enumerated separately; they coincide here. No winner was chosen by a random tie-break.

{table(['Model','Scope','Candidate ID','Method','Saved layer','C','Exact maximum'],winner_rows)}

Saved layers are zero-based (`hidden_states[saved_layer + 1]`). The winning Qwen C=10 and Llama C=1 probes are the original raw-feature L2 logistic fits on 2,778 P15 atomic rows, with an unpenalized intercept and objective mean BCE plus squared weights divided by 2Cn. All original fitted method-specific preprocessing is retained. `head_provenance.json` preserves complete settings, original fitting producers and fit/parameter references for every candidate.

## Frozen comparison procedures

{table(['Model','Procedure','Allocation 11','Allocation 23','Allocation 37','Mean AUROC'],allocation_rows)}

S_all is original reduced-bank Compound selection using A; R_all is Joint training. Means are the ordinary means of three separately calculated AUROCs. Predictions are neither averaged nor concatenated. Their accepted E scores and metrics were independently reconstructed. Maximum-based differences above are descriptive point differences only: no fixed-winner confidence interval is attached, and no bootstrap analysis was run.

## Verification and independent audit

- Complete grid verified against the B25 bank manifest, saved fits, exact parameter hashes and original D eligibility. The independent audit separately reconstructed the 10-settings-per-layer grid and atomic-D gate.
- Accepted E loader, model provenance, FP16 storage, saved-layer indexing, text identities and logical row ordering verified. Every candidate is bound to the same 2,700-row identity/label/order hash. All predictions are finite.
- All {execution['overlap_heads']} overlapping reduced-bank heads reproduce the accepted scores **exactly on all 7,200 bare rows**, including the boundary subset. {execution['overlap_metric_rows']} accepted endpoint rows were checked. The independent audit rechecked all {audit['overlapping_reduced_heads_independently_verified']} boundary score vectors against the original accepted predictions.
- All 600 AUROCs independently recomputed from saved scores using a Mann–Whitney average-rank calculation, separate from the production weighted pair-count implementation. Maximum absolute discrepancy: {audit['maximum_AUROC_error']:.3g}.
- Both maxima, every exact/tolerance winner, all 12 frozen allocation AUROCs, four means, and every reported descriptive gap independently reconstructed.
- Six focused tests passed: exact score ties, exact/tolerance maximum ties, eligibility filtering, missing/duplicate/substituted/invalid candidates, row/label misalignment, and nonfinite or unscored candidates. No unrelated suite or complete historical evaluation was rerun.
- No fitting, cross-validation, re-extraction, GPU work, full-bank evaluation, wording sweep, score flipping, recalibration, ensembles, or new E thresholds.

## Lineage and source hashes

Reviewed base: `a8b4cd326daa243338dc7941c19d98ce75f5b77f`.
Analysis code commit used: `{execution['started']['analysis_code_commit']}`.
B25 delivery: `{execution['b25_delivery']}`; completed successor producer: `{execution['b25_producer']}`.
Original E extraction producer: `{execution['original_E_extraction_producer']}`.
Unchanged scientific-lock SHA-256: `{execution['scientific_lock_sha256']}`.
Training-representation manifest SHA-256: `{execution['training_representation_manifest_sha256']}`.
Original parameter producers are preserved: 62 heads from `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`, 23 from `b25581ca9c22d93c8c988c93c0c5491219d6cfec`, and 515 from `5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4`.

The reviewed paper README led to the accepted scientific evaluation and independent audit handoffs. Both completion hashes and the successful audit companion were checked. `input_hashes.json` records source manifests, heads, preprocessing archives, original scores, raw-loader receipts, representation files and code hashes. `support_receipt.json` pins the E support. Historical access records identify accepted inputs; the present user request and `analysis_spec.json` authorize this separate analysis.

## Runtime and reproduction

Local macOS, one analysis process, one BLAS/OpenMP thread, protected by `/tmp/truth-probing-cpu-exclusive.lock`. Final scoring/verification wall time: **{runtime['wall_seconds']:.3f} seconds**; process peak resident memory: **{runtime['peak_rss_bytes']:,} bytes ({runtime['peak_rss_bytes']/1024**3:.3f} GiB)**. These are measured by a monotonic clock and macOS `getrusage`, respectively. Independent audit: {audit['wall_seconds']:.3f} seconds and {audit['peak_rss_bytes']:,} peak RSS bytes. These measurements cover each script's analysis/audit function and whole-process peak memory, not Git publishing.

The preliminary run completed scientific scoring but had a shadowed elapsed-time variable. Its outputs remain intact in the external run-v1 directory, and `superseded_run.json` records why its timing was rejected. The corrected implementation was committed and the unchanged scientific calculation was executed in fresh run-v2. `rerun_consistency.json` checks that all scores and numerical results are unchanged.

Large arrays remain outside Git at `{execution['external']}`. The directory is immutable for this run; a reproduction must choose a new external destination and a fresh small-output directory containing a copy of `analysis_spec.json`. From the new worktree:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B -m unittest tests.test_posthoc_reduced_bank_e -v
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/posthoc_reduced_bank_e.py --external {execution['external']}
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/audit_posthoc_reduced_bank_e.py
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/report_posthoc_reduced_bank_e.py
```

The listed scoring command is the command actually used; it deliberately refuses to overwrite its outputs. Relocated audit/report runs accept `--results`. The original manifests contain absolute artifact roots; all relocated inputs must still match their original hashes. The final delivery SHA is supplied with the verified remote handoff rather than self-embedded in this report.

## Files

- `candidate_results.csv`: all 600 candidates, methods/layers/C/settings, original eligibility, AUROC, producer/hash identities and scoring status.
- `summary.csv`, `winning_candidates.csv`, `fixed_procedures.csv`: both maxima, all ties, descriptive gaps and each allocation.
- `analysis_spec.json`, `workspace.json`, `execution.json`, `audit.json`, `focused_tests.json`, `input_hashes.json`, `head_provenance.json`: specification, source lineage, measured runtime and verification evidence.
- `controls.tex`: proposed Controls paragraph. No manuscript file was edited.
'''
    (out/'REPORT.md').write_text(report)
    tex=r'''\paragraph{Post-hoc test-set maximum.}
As an additional descriptive control, we scored all 280 Qwen and 320 Llama
frozen reduced-bank candidates on the 2,700 bare OR-boundary rows of final
test $E$. The post-hoc test-set maxima were @QMAX@ and @LMAX@, respectively;
applying the original atomic-$D$ eligibility rule gave the same maxima.
These exceed the unchanged compound-selection means by @QSEL@ and @LSEL@,
while the unchanged joint-training means remain higher by @QGAIN@ and @LGAIN@.
Procedure means average three allocation-specific AUROCs, not predictions.
E labels identify these winners: the maxima are optimistic descriptive
references on this observed test set, not independently evaluated selection
procedures or bounds on future-population performance. They concern only the
finite tested bank, not every possible atomic-trained probe. We attach no
fixed-winner confidence intervals to the maxima or their gaps. This analysis
was added after the original frozen evaluation and changes no selected head,
primary comparison, or scientific lock.
'''
    for key,value in {'QMAX':q['maximum'],'LMAX':l['maximum'],'QSEL':q['maximum_minus_S_all'],
                      'LSEL':l['maximum_minus_S_all'],'QGAIN':q['R_all_minus_maximum'],'LGAIN':l['R_all_minus_maximum']}.items():
        tex=tex.replace('@'+key+'@',fmt(value))
    (out/'controls.tex').write_text(tex)
    inventory={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json'}
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
    print(out/'REPORT.md')


if __name__=='__main__':main()
