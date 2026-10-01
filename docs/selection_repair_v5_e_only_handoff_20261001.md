# Reviewed E-only v5 handoff — 2026-10-01

The user reports reviewed mac 1 correction
`bb8bc51a864a5702f06ef23f923e11af97710c09` as an E-only update. Per that
handoff, v5 preserves train, D, A15/A10, T_C and actual P sampler identities.
This notification is recorded without fetching, pulling or merging v5, and
without accessing E data. Its identity-preservation claims have not been
independently revalidated in this task branch; exact compatibility checks belong
to the later clean-checkpoint integration.

## Completed v4 checkpoint

The notification arrived after the canonical refresh and capacity pilot had
completed and been pushed at
`b86500d22d5b6d0a1d6ba7db599061937def5197` on
`t2-capacity-pilot-20261001`. Execution remains local macOS,
`MacBook-Pro-7.lan`. No fit was interrupted or restarted.

All results remain attributed to **v4**, reviewed correction
`fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992`: 154 canonical summaries and
72 unique pilot configurations (70 new fits, two exact canonical reuses), zero
failed fits, and 1,260 pilot summaries. Existing adoption, inputs, configurations,
mappings, fits, scores, receipts and reports are unchanged. These are not fits
newly performed under v5.

The pilot retains its 14 review triggers and its recommendation to hold the design
freeze and consider a separately authorized nested-P10 comparison. This E-only
handoff changes neither that result nor A membership. P/A remains unfrozen;
no P10/P20, production-bank, compound-control or repair fitting is started.

Bound completion receipt hashes:

- V4 canonical sensitivity: `255d2267d4ffa20ac3fadfd72c0283fc86f6fe6da7676f36d7ec64e2a0c2d629`
- V4 capacity pilot: `9512909590dc8e6b2b34303424b8e4009ad1c3b78341c3b94887f607c46eb964`

## Later precision-check proposal

The following are the user-supplied current **E proposal** counts, not a frozen
allocation or an independently inspected E dataset:

| Topic | Proposed entities |
|---|---:|
| Animals | 16 |
| Cities | 146 |
| Elements | 18 |
| Inventors | 11 |
| Spanish | 34 |
| Total | 225 |

The supplied degree-four capacity is **450 pairs / 7,200 binary rows**. The
arithmetic is 225 × 4 / 2 pairs and 16 variants per pair. Shared entities and
pair variants make these rows dependent: **7,200 rows are not 7,200 independent
observations**. These capacity counts alone establish neither adequate precision
nor a final-test sample size justification.

At the later precision check, bind the exact reviewed entity and pair proposal,
account for shared-entity and within-pair dependence, and evaluate topic-level
precision, especially the smaller topics. Do not substitute raw binary-row count
for independent information. No E prediction, extraction, fitting or precision
simulation is performed by this handoff.

Integrate the reviewed branches only at the later clean checkpoint, preserving
v4 provenance and validating exact identities rather than count equality.
