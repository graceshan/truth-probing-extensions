# Seven-claim source closure and correction-overlay recommendation

Reviewed starting commit: `b8e27508f20bab896a46f448081f43ed32d919c4` on
`clean-eval-protocol`, mac 1 (`Darwin`), in the intended
`/Users/apple/projects/truth-probing-extensions` checkout. Fetch confirmed HEAD
and origin still agreed; no pull was needed. No mac 2 branch or worktree was
merged or modified. The 13 unrelated untracked files remain byte-identical.

**Recommendation: review the separately versioned `candidate_overlay_v2` as
the conservative correction overlay. It is not adopted or made a production
default by this task.** Keep the five corroborated conflicting pairs excluded;
classify Walton/US and Bengio/US as unresolved and quarantine their pairs.
Unresolved here means corroboration was insufficient, not that residence was
disproved. All seven remain nonadmitted and no original labels are changed.

## Evidence and dispositions

`source_closure_v1/policy.json` and its exact seven-case `queue.json` were saved
and hashed before new retrievals. The existing lived-in policy remains in force:
an established home, settlement or documented residential study/employment
posting; nationality, visits and an institution's location alone do not suffice.
No new duration threshold, country mapping or alias merge is introduced.

| Claim and original affirmative ID | Closure classification | Relevant inspected support | Pair disposition |
|---|---|---|---|
| Walton/US, `inventors:2` | Unresolved; not a confirmed error | Saved secondary account reports a two-year factory-startup stay. Accessible Vivian memoir passages concern inventions and a Brighton house; they do not corroborate the US stay. ACS historical article blocked. | Excluded → quarantined |
| Marconi/UK, `inventors:68` | Corroborated label conflict | Saved English Heritage London inscription explicitly says “lived here in 1896–1897.” | Remains excluded |
| Kapany/UK, `inventors:139` | Corroborated label conflict | UCSC obituary says he went to England for graduate school, describes London research in 1953 and his 1955 doctorate. This is relocation for study, not location-only inference. | Remains excluded |
| Göbel/US, `inventors:233` | Corroborated label conflict | Springe municipal history explicitly describes emigration to New York in 1848. Saved contemporary Henry Goebel patent independently identifies New York locality. | Remains excluded |
| Bengio/US, `inventors:234` | Unresolved; not a confirmed error | Saved secondary biography reports MIT/Bell Labs postdocs. Retrieved CIFAR text covers Canadian appointments; university CV paths failed, university index supplied no text, and ACM was blocked. Residential nature was not sufficiently corroborated. | Excluded → quarantined |
| Ericsson/US, `inventors:240` | Corroborated label conflict | Church's historical biography records Astor House residence, a Franklin Street office/home and a recalled 1887 visit to his Beach Street home in New York. | Remains excluded |
| Brill/US, `inventors:377` | Corroborated label conflict | Saved NIHF biography specifies California employment with concurrent graduate classes and a USC degree. | Remains excluded |

Sources with new substantive corroboration:

- [UC Santa Cruz: Kapany obituary](https://news.ucsc.edu/2020/12/narinder-kapany-in-memoriam.html), paragraph beginning “Born in India…”.
- [Stadt Springe: Heinrich Göbel](https://www.springe.de/portal/seiten/heinrich-goebel-900000162-24600.html?rubrik=900000002), “Mehr Informationen zu Heinrich Göbel,” emigration passage. The city's discussion of disputed invention legends is not used as invention evidence.
- [Church: The Life of John Ericsson, Volume I](https://archive.org/stream/lifeofjohnericss011402mbp/lifeofjohnericss011402mbp_djvu.txt), chapter “Removal to the United States,” pp.111,114–115. Scanned edition 1906, copyright 1890; includes personal recollections and correspondence. OCR imperfections are retained and disclosed.

The saved English Heritage, NIHF and patent response bytes were reinspected and
bound to their original v1 hashes. `evidence.json` contains titles, URLs, retrieval
dates, excerpts/locators, authority, actual support and limitations. The original
secondary leads remain in v1 and are referenced, not erased or recast as false.
All judgments are **Codex-assisted external review, not human verification**.

## Bounded coverage and preservation

This closes dispositions for exactly the seven new v1 conflict claims. It does
not re-adjudicate the other 35 members of the 42-case queue or expand the 140-row
screen into a lifetime biography audit. Each of the five secondary-dependent
claims received three discovery queries; Google returned script-only pages and
DuckDuckGo returned bot challenges. No search snippets were used as evidence.
Direct source attempts numbered Walton 4, Kapany 2, Göbel 4, Bengio 5 and Ericsson
4, within the declared six-page bound. This includes failed/irrelevant attempts,
not a claim that every page supplied evidence. Marconi and Brill used their saved
institutional evidence. Discovery failures, timeouts, a wrong patent-number lead,
and narrower/noncorroborating material are explicitly recorded in the retrieval
log. We stopped within the bound rather than treating inaccessible evidence as
confirmation.

Every file in `candidate_overlay_v1`, including its 42-case queue, reviews and
receipt, remains byte-identical. The original 30-case sample, draw order and
outcomes remain **4 supported false / 2 supported true / 24 unresolved**. Closure
reviews add zero cases to that denominator. Franklin/France remains separately
ineligible as a historical negative candidate, with no invented source pair.

The successor manifest preserves every v1 column and appends `successor_status`,
`closure_decision_ref` and a pending-review marker. Consumers reviewing v2 must
use `successor_status`; `candidate_status` remains the historical v1 value.
There are exactly four status changes, all train rows: `inventors:2`,
`neg_inventors:2`, `inventors:234`, `neg_inventors:234`.

| Partition | Admitted v1 and v2 | Excluded v1 → v2 | Quarantined v1 → v2 |
|---|---:|---:|---:|
| Train | 3,048 | 24 → 20 | 72 → 76 |
| D | 1,012 | 4 → 4 | 24 → 24 |
| E | 1,002 | 8 → 8 | 18 → 18 |
| **Total** | **5,062** | **36 → 32** | **114 → 118** |

The full ordered admitted identity table is byte-identical when projected from
v1 and v2. Its hash is recorded twice in `closure_receipt.json`, and its content
is saved as `candidate_overlay_v2/admitted_identities.csv`. It includes source
IDs/hashes, statement hashes, labels, partitions, original entity IDs, reviewed
person keys and paired IDs. No row moves partitions. Simjian remains quarantined
across train/D; Farnsworth's two original train entity IDs retain one person key.

## Corrected roles and factual-readiness requirements

This section supersedes the role interpretation in the earlier v1 capacity
discussion; that historical document itself remains unchanged.

- **T_C may overlap P and A; T_C must exclude D/E.** It is not required to be
  disjoint from both P and A. No role assignments are made here.
- **20 audited usable T_C entities per topic is an outer-training target.**
  It does not require 20 D or E entities. The smaller D/E animal and element
  pools therefore do not block that training target.
- Extra T_C facts can come from P without enlarging A. The available outer-train
  positive-bearing bounds remain 450 cities, 106 Spanish, 126 inventors,
  57 elements and 48 animals. These are availability bounds, not factual audit
  certificates or reserved sets.
- Keep **15/topic as the planned reservation pilot**, **10 as fallback**, and
  **20 only for a deliberate B=50 commitment**. No A/P entities are selected or
  reserved, and no B=50 commitment is made by this package.

Retained source labels, externally reviewed negatives, and fully audited
same-entity positive/negative fact pairs are separate evidence tiers. The
membership change is zero, so factual availability and audited pair counts are
unchanged: train has two completed inventor source pairs (Edison, Maxwell),
E one (Porro), and no completed pairs in the other topics under this ledger's
independent-review standard. The v2 capacity table updates restriction categories
only; all factual availability fields match v1. “Audited” still denotes the
documented Codex-assisted review, not human certification.

The recovered registry evidence remains intact, including **34 externally reviewed
Spanish D negatives**. Those are D evidence, not permission to transfer D facts
into training. Spanish train has 106 positive-bearing keys but only four source
same-key positive/negative pairs. Before any proposed A/T_C facts are used, both
affirmative polarities must have exact, compatible identity and factual support
on each relevant training entity. Review same-word negatives and positives for
polysemy, dialect, spelling/diacritics and translation direction; false-only words
cannot borrow another word's true fact, and negation cannot create the missing
independent affirmative fact. No new facts are generated here.

For an eventual 20/topic T_C target, current outer-train bounds suffice
conditionally, but training inventories still need 20 completed pair audits per
non-inventor topic and at least 18 additional completed inventor pair audits.
Overlapping P/A facts may satisfy these requirements after review; they need not
be 20 newly reserved adaptation entities. For the 15/topic pilot or 10 fallback,
review the actually proposed A and T_C fact sets under the subsequently approved
role plan. Passing a capacity bound does not guarantee factual review succeeds.

## Recommendation and remaining decisions

Recommend accepting v2's **conservative nonadmission membership and corrected
classification**, subject to owner review of this evidence package. Walton and
Bengio may remain quarantined without forcing a binary factual judgment; their
unresolved status does not itself require expanding this bounded audit. The
other existing quarantines also remain in force.

Correction-overlay adoption still requires explicit review/acceptance of the
five corroborated judgments and the conservative treatment of unresolved claims.
This task does not make that decision or promote a default. Separately, use in
experiments requires factual validation of proposed A/T_C same-entity fact pairs
and an approved role/budget plan. Neither role overlap nor the capacity inventory
removes those requirements. No activation compatibility is asserted; any later
reuse requires model/revision, layer, token-position, source-index and sidecar
checks. No fitting, activation access, extraction, compounds or final-test
predictions/metrics occurred.

## Validation

```bash
/tmp/clean-extraction-venv/bin/python -B scripts/48_close_selection_repair_sources.py
/tmp/clean-extraction-venv/bin/python -B scripts/48_close_selection_repair_sources.py --check-only
/tmp/clean-extraction-venv/bin/python -B scripts/47_build_selection_repair_candidate.py --check-only
/tmp/clean-extraction-venv/bin/python -B scripts/46_audit_selection_repair_sources.py
/tmp/clean-extraction-venv/bin/python -B data/clean_protocol/selection_repair_v1/source_review_v1/validate.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_selection_repair_source_closure.py tests/test_selection_repair_source_audit.py tests/test_selection_repair_candidate_overlay.py data/clean_protocol/selection_repair_v1/source_review_v1/test_validation.py
git diff --check
```

Result: **68 tests plus 17 subtests passed**. Relevant existing T2A, T2B and v1
validators pass. Tests cover identity/pair mutations, evidence requirements,
sample preservation, exact admission equality, count reconciliation, alias
handling, Spanish evidence retention, deterministic generation and overwrite
refusal. Existing tests, including the test file whose portable fix belongs to
mac 2, are unchanged. No scientific experiment was run.
