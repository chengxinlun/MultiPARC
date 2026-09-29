from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.isotonic import IsotonicRegression


def cumulative_mc_stats_welford(file_dir, expected_shape, pattern="*.pt", sort_key=None, track_scalar="mean_std"):
    """
    Incrementally computes running mean/std over saved inference files,
    without ever stacking more than one sample in memory at a time.

    Args:
        file_dir: path to where the mc dropout inferences are stored
        expected_shape: tuple of the expected shape of the tensor, excluding batch dimension
        pattern: file patterns. Default is '*.pt'.
        sort_key: sort file according to key. Default is None.
        track_scalar: wheather to track the mean of std ('mean_std', default) or the max of std
            ('max_std').

    Returns:
        final_mean, final_std: tensors of shape (-1, *expected_shape),
            the full-field stats using all files.
        scalar_curve: list of length n_files, a scalar summary of the
            running std at each cumulative step, for plotting convergence.
        n_samples: list of sample counts, same length as scalar_curve.
    """

    files = sorted(Path(file_dir).glob(pattern), key=sort_key)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {file_dir}")

    mean = None
    M2 = None  # sum of squared deviations from the running mean
    scalar_curve = []
    n_samples = []

    for k, f in enumerate(files, start=1):
        x = torch.load(f).reshape(-1, *expected_shape)

        if mean is None:
            mean = torch.zeros_like(x)
            M2 = torch.zeros_like(x)

        delta = x - mean
        mean += delta / k
        delta2 = x - mean
        M2 += delta * delta2

        if k >= 2:
            running_std = torch.sqrt(M2 / (k - 1))
            if track_scalar == "mean_std":
                scalar_curve.append(running_std.mean(dim=(0,2,3)))
            elif track_scalar == "max_std":
                scalar_curve.append(running_std.max(dim=(0,2,3)))
            n_samples.append(k)

        del x  # drop reference so it can be freed before next file loads

    final_std = torch.sqrt(M2 / (len(files) - 1))
    return mean, final_std, scalar_curve, n_samples


def fit_isotonic_calibrator(abs_error, std, z_grid=None, eps=1e-12):
    """
    Fits a post-hoc calibrator mapping empirical coverage to the z-score
    that actually achieves it, using MC dropout std as the uncertainty proxy.
    
    For a dense grid of z-values (0.001 to 6.0 by default, spanning near-zero
    to near-full coverage), computes empirical coverage (fraction of points
    with abs_error <= z * std), then fits an isotonic regression from
    coverage -> z. At application time, querying this mapping at a target
    confidence level p gives the z actually needed for that coverage;
    dividing by the nominal Gaussian z_p yields a multiplier to rescale raw
    std (see make_multiplier_fn). The wide z_grid avoids the flat-plateau
    clipping seen when the grid only spans the reporting confidence levels.

    Returns the fitted IsotonicRegression object directly (not a closure),
    so it can be pickled and reloaded later.

    Args:
        abs_error: true absolute errors from a held-out calibration set,
            any shape, flattened internally.
        std: predicted std (e.g. from MC dropout), same shape as abs_error.
            Clipped to at least eps.
        z_grid: z-values to evaluate coverage at for fitting. Widen the
            upper bound if coverage doesn't approach 1.0 by 6.0 for your data.
        eps: floor applied to std to avoid division by zero.
    
    Returns:
        Fitted sklearn.isotonic.IsotonicRegression mapping coverage -> z.
    """

    abs_error = abs_error.flatten()
    std = np.clip(std.flatten(), eps, None)

    if z_grid is None:
        z_grid = np.linspace(0.001, 6.0, 200)

    coverage_grid = np.array([(abs_error <= z * std).mean() for z in z_grid])

    inverse = IsotonicRegression(increasing=True, out_of_bounds="clip")
    inverse.fit(coverage_grid, z_grid)
    return inverse


def make_multiplier_fn(inverse):
    """
    Rebuilds the multiplier closure from a fitted IsotonicRegression.

    Args:
        inverse: fitted sklearn.isotonic.IsotonicRegression mapping

    Returns:
        Callable that maps probability to the multipiler for uncertainty calibration
    """
    def multiplier(p):
        z_p = stats.norm.ppf((1 + p) / 2)
        z_actual_needed = inverse.predict([p])[0]
        return z_actual_needed / z_p
    return multiplier


def compute_coverage_curve(abs_error, std, confidence_levels=None, multiplier_fn=None, eps=1e-12):
    """
    Computes empirical coverage at each nominal confidence level.

    Args:
        abs_error, std: flattened or broadcastable arrays.
        confidence_levels: nominal levels to evaluate, default 0.05..0.95 step 0.05.
        multiplier_fn: optional callable, p -> scalar multiplier applied to std
            at that confidence level (e.g. from fit_isotonic_calibrator).
            If None, std is used as-is (raw) or pre-scaled by the caller
            (e.g. temperature scaling, apply before calling this).

    Returns:
        dict with confidence_levels, empirical_coverage, ece
    """
    abs_error = abs_error.flatten()
    std = np.clip(std.flatten(), eps, None)

    if confidence_levels is None:
        confidence_levels = np.arange(0.05, 1.0, 0.05)

    empirical_coverage = []
    for p in confidence_levels:
        z_p = stats.norm.ppf((1 + p) / 2)
        m_p = multiplier_fn(p) if multiplier_fn is not None else 1.0
        covered = (abs_error <= z_p * (std * m_p)).mean()
        empirical_coverage.append(covered)

    empirical_coverage = np.array(empirical_coverage)
    ece = np.mean(np.abs(empirical_coverage - confidence_levels))

    return {"confidence_levels": confidence_levels, "empirical_coverage": empirical_coverage, "ece": ece}


def plot_reliability_diagram(results, label="MC dropout", ax=None):
    """
    Plots a reliability diagram (empirical coverage vs. nominal confidence
    level) from one or more coverage curves, for visually comparing
    calibration quality.

    Draws the diagonal "perfect calibration" reference line, then one curve
    per entry in `results`, each labeled with its ECE. Accepts either a
    single dict from compute_coverage_curve (using `label`), or a list of
    (results, label) tuples to overlay several curves (e.g. raw vs.
    temperature-scaled vs. isotonic) on the same axes for direct comparison.

    If `ax` is provided, plots onto it without creating or showing a new
    figure, letting the caller compose this into a larger figure (e.g. a
    grid of per-channel diagrams). Otherwise creates its own figure and
    displays it.

    Args:
        results: a single dict as returned by compute_coverage_curve (with
            "confidence_levels", "empirical_coverage", "ece" keys), or a
            list of (results_dict, label) tuples to overlay multiple curves.
        label: legend label used when `results` is a single dict. Ignored
            if `results` is a list of (results, label) tuples.
        ax: optional matplotlib Axes to plot onto. If None, a new figure
            and axes are created and displayed.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    if isinstance(results, dict):
        results = [(results, label)]

    for res, lbl in results:
        ax.plot(res["confidence_levels"], res["empirical_coverage"], marker="o",
                markersize=4, label=f"{lbl} (ECE={res['ece']:.4f})")

    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()
