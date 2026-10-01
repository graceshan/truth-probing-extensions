# T1: recovery of the audited development benchmark

Recovered on 2026-10-01 from the existing local working tree at
`7a8561c3ef58fd28836cec3c52594b4ec9b4d904` on `clean-eval-protocol`.
After fetching, origin and local HEAD agreed; origin identified
`graceshan/truth-probing-extensions`. This is historical recovery, not a new
label/alias cleanup. No recovered code, configuration, judgment, evidence, or
historical metadata bytes were edited.

## Preservation and scope

Before project changes, a backup was created outside the checkout at
`~/truth-probing-backups/t1-recovery-20261001T165637Z/`. It contains all 105
modified/untracked files, including unrelated work and ZIPs, plus seven tracked
source-audit files: 112 files / 61,479,974 bytes. It also contains Git status,
HEAD, branch/remotes, reflogs, binary diffs, and an all-ref Git bundle. Every
restored file matched its recorded size and SHA-256; the bundle verified.

- File-manifest SHA-256: `14d24a246fa32e9ffe2ba5ba8470ec57635f51a2b0f01006812870e33b9bdf0b`
- Working-files archive SHA-256: `cbdc308e4d8f75caa58384a58bf5746f1040df803aafa5543e88f8fe84f4c862`

The recovered set is 92 existing files, plus this note:

- The modified `src/clean_compounds.py` and seven modules: generalization
  protocols, validated negatives, negative-review batches, inventor-country
  semantics/audit, inventor-source consistency, and validation-compound generation.
- Scripts 22–27; four configs for generalization, initial negatives,
  single-country negatives, and the development/validation benchmark.
- Six existing corresponding test files and six existing documentation files.
- All 43 files under `data/clean_protocol/validated_negatives/v1/`, including
  TRAIN/VALIDATION registries, intermediate snapshots, review batches, imported
  judgments, evidence notes, and verification/provenance receipts.
- All 19 files in the inventor-single-country and inventor-source-consistency
  audit directories.

Existing tracked dependencies remain unchanged: `src/data.py`,
`src/entity_partitions.py`, `src/negative_audit.py`, synthetic compound fixtures,
entity manifests, source CSVs, and the original source audit. Import-dependency
coverage was checked against the recovered set and tracked files. Newer compound
transfer code/config/tests, ZIPs, caches, and generated outputs were not staged.

The local `clean_compounds.py` matches the historical benchmark's producer hash
`7f4c5643db07775c8f7bf5ae1b8ed3b81ed4dc0238d23a2d830178bacf8ae846`;
the pre-recovery committed version did not. All eight recorded producer-code
hashes match the recovered implementation. Accepted judgments, rejected
judgments, unverified candidates, and supporting evidence retain their original
distinct statuses. The pinned v5 registry retains 174 source-supported negatives,
88 external acceptances, one rejection, and 8,470 unverified candidates.
No external factual claim was independently re-reviewed during recovery.

## Validation

Using the existing `/tmp/clean-extraction-venv/bin/python` environment, the nine
focused test files passed: **81 tests and 42 subtests**. These cover the six
recovered test modules plus `test_clean_compounds.py`, `test_negative_audit.py`,
and `test_entity_partitions.py`. Tests use synthetic fixtures. The initial
source-only scratch run passed 80 tests; its Git-snapshot preservation test
failed because that export had no `.git` directory. The complete unchanged
suite then passed in the actual checkout. No implementation/test edits were
made to resolve that environment limitation.

The established command was run in an isolated scratch copy containing the
committed dependencies plus exact recovered bytes, with the original relative
paths and configuration. Historical benchmark outputs were not copied there:

```sh
python -B scripts/27_generate_validation_compound_benchmark.py \
  --config config/clean_protocol/entity_disjoint_development_validation_v1.json
```

It reproduced 8,384 rows, 524 pairs, 262 entities and degree four. All four CSVs
and `metadata.json` were byte-identical to the original checkout, including the
recorded producer hashes. No missing prerequisite or reproduction discrepancy
remained. Original benchmark files were never overwritten.

| Reproduced file | SHA-256 |
|---|---|
| `development_validation_compounds.csv` | `96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94` |
| `development_validation_pairs.csv` | `e59ce01242e82ff88c249766146ba89e8f897a2573ed1bc9b1c57eb9516aad61` |
| `development_validation_entity_degrees.csv` | `395d34f8a285567fc1583e4492a827c019eea865e1b11d3857909f8838f3e702` |
| `development_validation_selected_negative_facts.csv` | `c245e9a80f81d6f515de27d48f4e5d55438bd1c0df119af4ad428167a3e14b8a` |
| `metadata.json` | `5b0c7b885710b7d41f49a4eea71e5bd042050f0eb14646787927f1a76b47feeb` |

The historical loader hashes mixed-source files and filters split/entity
membership before factual use. Recovery did not open final-test candidate
queues, generate real final-test examples, inspect predictions, extract model
activations, or compute probe scores. Validation logs and dependency/reproduction
receipts are retained with the external backup, not as new scientific outputs.
