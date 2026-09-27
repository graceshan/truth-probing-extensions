"""Original atomic probe formulas. No data loading or validation selection here."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import eigh
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.diff_means import diff_of_means_direction

METHODS = ("l2_logistic", "difference_of_means", "mass_mean_covariance", "burger_t_g", "ttpd")
BURGER_TOPICS = ("cities", "sp_en_trans", "inventors", "animal_class", "element_symb")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def validate_training(X, y):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    check(X.ndim == 2 and y.shape == (len(X),) and X.shape[1] > 0, "invalid training dimensions")
    check(np.isfinite(X).all() and np.array_equal(np.unique(y), [0, 1]), "training requires finite data and both binary classes")
    return X, y


@dataclass
class FittedMethod:
    method: str
    parameters: dict
    details: dict

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        check(X.ndim == 2 and X.shape[1] == len(self.parameters["coef"]), "invalid prediction dimensions")
        check(np.isfinite(X).all(), "nonfinite prediction input")
        if self.method == "ttpd":
            # Faithful to the authors: polarity COEFFICIENT projection, not its
            # probability/intercept, and not t_P or a supplied form indicator.
            features = np.column_stack((X @ self.parameters["t_g"], X @ self.parameters["polarity_coef"]))
            return features @ self.parameters["head_coef"] + self.parameters["head_intercept"].item()
        return X @ self.parameters["coef"] + self.parameters["intercept"].item()


def difference_of_means(X, y):
    X, y = validate_training(X, y)
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    check(np.linalg.norm(mu1 - mu0) > 0, "zero difference-of-means direction")
    direction = diff_of_means_direction(X, y)
    check(direction @ (mu1 - mu0) > 0, "difference-of-means sign error")
    return FittedMethod("difference_of_means", {"coef": direction, "intercept": np.array(0.),
                        "mean_false": mu0, "mean_true": mu1, "raw_direction": mu1 - mu0},
                        {"formula": "unit(mean_true - mean_false)", "fit_rows": len(X),
                         "sign_orientation": "true-minus-false; no validation-dependent flip"})


def covariance_mass_mean(X, y, atol=1e-3):
    """Marks--Tegmark MMProbe(iid=True): Sigma^+ (mu1-mu0).

    Sigma = sum within-class residual outer products / N (not total covariance,
    not N-2, no shrinkage). Original torch pinv uses hermitian=True, atol=1e-3;
    explicit positive atol gives rtol=0. Apply its spectral inverse to the
    direction without constructing a full inverse. Float64 symmetric eigh is
    an exact algebraic implementation of the same thresholded pseudoinverse.
    """
    X, y = validate_training(X, y)
    check(atol == 1e-3, "original MM absolute pseudoinverse cutoff is fixed at 1e-3")
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    delta = mu1 - mu0
    residuals = X - np.where(y[:, None] == 1, mu1, mu0)
    covariance = residuals.T @ residuals / len(X)
    eigenvalues, vectors = eigh(covariance, check_finite=False, driver="evr")
    keep = np.abs(eigenvalues) > atol
    direction = vectors[:, keep] @ ((vectors[:, keep].T @ delta) / eigenvalues[keep])
    check(delta @ direction >= -1e-10 * max(1., np.linalg.norm(delta) * np.linalg.norm(direction)),
          "covariance-adjusted direction has reversed sign")
    return FittedMethod("mass_mean_covariance", {"coef": direction, "intercept": np.array(0.),
                        "mean_false": mu0, "mean_true": mu1, "raw_direction": delta,
                        "covariance_eigenvalues": eigenvalues, "pinv_atol": np.array(atol)},
                        {"formula": "pinv(within_class_covariance, atol=1e-3, rtol=0) @ (mu1-mu0)",
                         "covariance_divisor": len(X), "covariance_rank_retained": int(keep.sum()),
                         "fit_rows": len(X), "intercept": "none in upstream iid MM readout",
                         "sign_orientation": "PSD inverse of true-minus-false; no sign flip"})


def learn_truth_directions(X, y, polarities, datasets):
    """Bürger eqs. 3--5 and probes.learn_truth_directions, including dataset means."""
    X, y = validate_training(X, y)
    p, datasets = np.asarray(polarities), np.asarray(datasets)
    check(p.shape == y.shape and np.array_equal(np.unique(p), [-1, 1]), "both polarities required")
    check(datasets.shape == y.shape, "dataset identity dimensions differ")
    names = np.unique(datasets)
    centered, means = X.copy(), []
    for dataset in names:
        mask = datasets == dataset
        check(len(np.unique(p[mask])) == 1, "center each topic/form dataset separately")
        mean = X[mask].mean(0)
        means.append(mean)
        centered[mask] -= mean
    tau = 2 * y - 1
    design = np.column_stack((tau, tau * p))
    gram = design.T @ design
    check(np.linalg.matrix_rank(gram) == 2, "truth/polarity OLS design is singular")
    solution = np.linalg.solve(gram, design.T @ centered)
    tg, tp = solution
    # Residualize the truth regressor against the interaction regressor: its
    # covariance with the fitted t_G readout must have the true-positive sign.
    partial_truth = tau - design[:, 1] * (design[:, 1] @ tau) / (design[:, 1] @ design[:, 1])
    alignment = float(partial_truth @ (centered @ tg))
    check(alignment > 0, "general-truth direction is degenerate or incorrectly oriented")
    params = {"t_g": tg, "t_p": tp, "dataset_names": names.astype(str), "dataset_means": np.stack(means),
              "ols_gram": gram, "coef": tg, "intercept": np.array(0.)}
    return FittedMethod("burger_t_g", params,
                        {"formula": "solve(Z.T@Z, Z.T@(X - mean_of_training_dataset)); Z=[tau,tau*p]",
                         "fit_rows": len(X), "inference": "raw activation dot t_G; no validation centering",
                         "truth_encoding": "false=-1,true=+1", "polarity_encoding": "negated=-1,affirmative=+1",
                         "sign_orientation": "OLS truth coefficient, never flipped on validation",
                         "partial_truth_training_alignment": alignment})


def unregularized_logistic(X, y):
    """Original TTPD objective; increase optimizer budget only if it fails to converge."""
    attempts = []
    for iterations in (100, 10000):
        model = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs", tol=1e-4,
                                   max_iter=iterations)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        attempts.append({"max_iter": iterations, "n_iter": int(model.n_iter_[0]),
                         "convergence_warnings": convergence})
        if not convergence:
            check(np.array_equal(model.classes_, [0, 1]), "logistic class orientation error")
            return model, attempts
    raise RuntimeError("unregularized TTPD logistic fit did not converge; no regularization fallback")


def fit_ttpd(X, y, polarities, datasets, truth=None):
    X, y = validate_training(X, y)
    truth = truth or learn_truth_directions(X, y, polarities, datasets)
    polarity_labels = (np.asarray(polarities) == 1).astype(int)
    polarity_model, polarity_optimization = unregularized_logistic(X, polarity_labels)
    p_direction = polarity_model.coef_[0]
    polarity_alignment = float(p_direction @ (X[polarity_labels == 1].mean(0) - X[polarity_labels == 0].mean(0)))
    check(polarity_alignment > 0, "polarity classifier sign is not affirmative-positive")
    features = np.column_stack((X @ truth.parameters["t_g"], X @ p_direction))
    head, head_optimization = unregularized_logistic(features, y)
    coef = head.coef_[0, 0] * truth.parameters["t_g"] + head.coef_[0, 1] * p_direction
    params = dict(truth.parameters, coef=coef, intercept=head.intercept_[0],
                  polarity_coef=p_direction, polarity_intercept=polarity_model.intercept_,
                  polarity_classes=polarity_model.classes_, head_coef=head.coef_[0],
                  head_intercept=head.intercept_[0], classes=head.classes_)
    return FittedMethod("ttpd", params, {"fit_rows": len(X),
                        "formula": "unregularized LR([X@t_G, X@polarity_LR.coef])",
                        "polarity_projection_excludes_intercept": True,
                        "polarity_optimizer": polarity_optimization, "head_optimizer": head_optimization,
                        "polarity_training_alignment": polarity_alignment,
                        "sign_orientation": "truth LR classes [0,1]; polarity LR negated=0,affirmative=1",
                        "truth_fit": truth.details})


def balanced_burger_indices(rows, seed=0):
    """Original equal-dataset sampling, same row subset for each affirmative/negated pair."""
    check(set(rows.split) == {"train"}, "Bürger subsampling requires TRAIN only")
    grouped = {}
    for topic in BURGER_TOPICS:
        affirmative = rows.index[(rows.topic == topic) & (rows.form == "affirmative")].to_numpy()
        negated = rows.index[(rows.topic == topic) & (rows.form == "negated")].to_numpy()
        check(len(affirmative) == len(negated) > 0, "affirmative/negated TRAIN counts must match")
        a, n = rows.loc[affirmative], rows.loc[negated]
        check(np.array_equal(a.row_index, n.row_index) and np.array_equal(a.entity_id, n.entity_id),
              "paired row identities do not align")
        check(np.array_equal(a.label.to_numpy(), 1 - n.label.to_numpy()), "paired labels are not complementary")
        grouped[topic] = affirmative, negated
    size = min(len(pair[0]) for pair in grouped.values())
    rng, indices = np.random.RandomState(seed), []
    for topic in BURGER_TOPICS:
        a, n = grouped[topic]
        selected = rng.choice(len(a), size, replace=False)
        indices.extend(a[selected])
        indices.extend(n[selected])
    return np.asarray(indices, dtype=int)
