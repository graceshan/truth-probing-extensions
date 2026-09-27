# Atomic LR convergence audit

Run `python3 -B scripts/21_audit_atomic_convergence.py`. This audits only rows
marked `convergence_warning=True` in the existing validation grids. It does not
rerun the hyperparameter search or replace any selection artifact.

Each warned configuration is refit from scratch on the same manifest-assigned
train rows at the same layer and C. The only parameter change is
`max_iter=2000` to `max_iter=10000`. L2, lbfgs, intercept, tolerance, class weights,
activation conversion, labels, and BLAS thread count remain unchanged. The
runner verifies original source/cache metadata, manifest assignments, code and
configuration hashes, runtime versions, and saved probe/validation hashes.

Validation rows alone are scored. Test access remains locked; the audit runner
has no test-evaluation option. Any remaining convergence warnings are recorded
without further optimizer changes.

Outputs are separate under
`results/clean_protocol/atomic_lr_convergence_audit/{model}/`:

- `refits.csv`: model, layer, C, original iterations/AUROC/status, refit
  iterations/AUROC/status, absolute AUROC change, diagnostics and warning text.
- `refit_probes.npz`: the independently refitted coefficients and intercepts.
- `summary.json`: original/refit configurations and provenance, artifact hashes
  and modification times verified unchanged, test-access counters, and full-grid
  selection comparisons.

For the comparison, replace only the warned configurations whose refits converged,
then apply the original global rule within each model: highest pooled validation
AUROC, smaller C, lower layer. Unresolved fits retain their original values and
make the comparison explicitly provisional. No selected layer/C or coefficient
archive is updated by this audit.
