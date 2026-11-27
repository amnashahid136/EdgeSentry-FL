"""
dp_utils.py
-----------
Differential-privacy utilities for DP-FedAvg (CLO3: "security & privacy at
the edge"). Implements the standard DP-FedAvg recipe from McMahan et al.,
"Learning Differentially Private Recurrent Language Models" (ICLR 2018):

  1. Each client's model UPDATE (local_weights - global_weights) is clipped
     to a fixed L2 norm `clip_norm`, bounding how much any single client
     can move the aggregate (this bounds each client's sensitivity).
  2. The clipped updates are summed and calibrated Gaussian noise is added
     to the SUM (not to each client individually) -- the noise standard
     deviation is `noise_multiplier * clip_norm`.
  3. The noised sum is divided by the number of participating clients to
     get the new global update.

Because the privacy guarantee below assumes a fixed, data-independent
denominator, DP mode averages clients UNIFORMLY (1/N each) instead of
weighting by local sample count as plain FedAvg does -- weighting by
sample count would leak information about each client's dataset size and
complicate the sensitivity analysis.

Epsilon accounting: this module reports a simple, non-tight (epsilon,
delta)-DP bound using the analytic Gaussian mechanism (Dwork & Roth,
"The Algorithmic Foundations of Differential Privacy", Theorem 3.22),
composed loosely (additively, i.e. basic composition) across rounds. This
is meant to give a rough, honest sense of the privacy budget for a course
project -- it is a deliberately conservative *upper bound*, not a tight
one. A real deployment would use a proper moments/RDP accountant (e.g.
`tensorflow_privacy.compute_dp_sgd_privacy`) which composes many rounds
far more tightly (roughly sqrt(rounds) instead of linear in rounds).
"""
import math

import numpy as np


def flatten_weights(weight_list):
    """Concatenate a list of per-layer ndarrays into one flat vector."""
    return np.concatenate([np.asarray(w).reshape(-1) for w in weight_list])


def clip_update(update, clip_norm):
    """Clip a per-client update (list of ndarrays, one per layer, already
    = local_weights - global_weights) to a global L2 norm of `clip_norm`.

    Returns (clipped_update, pre_clip_norm) so callers can log how often /
    how hard clipping actually engaged (a healthy DP run clips most
    rounds; if it never clips, `clip_norm` is too loose to bound
    sensitivity, and the privacy accounting above is not valid).
    """
    flat = flatten_weights(update)
    norm = float(np.linalg.norm(flat))
    factor = min(1.0, clip_norm / (norm + 1e-12))
    clipped = [layer * factor for layer in update]
    return clipped, norm


def add_gaussian_noise(summed_update, noise_multiplier, clip_norm, rng):
    """Add isotropic Gaussian noise (per-coordinate std = noise_multiplier
    * clip_norm) to the SUM of already-clipped per-client updates. This
    mirrors the standard DP-SGD noise-addition step, applied here to
    federated client updates instead of per-example gradients."""
    noise_std = noise_multiplier * clip_norm
    return [
        (layer + rng.normal(0.0, noise_std, size=layer.shape)).astype(layer.dtype)
        for layer in summed_update
    ]


def per_round_epsilon(noise_multiplier, delta=1e-5):
    """Analytic Gaussian mechanism bound: releasing one clipped-sensitivity
    query with noise std = noise_multiplier * sensitivity is
    (epsilon, delta)-DP for

        epsilon = sqrt(2 * ln(1.25 / delta)) / noise_multiplier

    (the sensitivity term cancels because noise_std is expressed as a
    multiple of it). This is the standard "basic" Gaussian-mechanism bound
    used throughout the DP-SGD literature before a tighter accountant is
    applied; it is a valid, if loose, upper bound.
    """
    return math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier


def composed_epsilon(noise_multiplier, rounds, delta=1e-5):
    """Loose (basic/sequential) composition of `rounds` independent noisy
    releases: epsilon_total = rounds * epsilon_per_round. This
    over-estimates the true privacy cost -- a tight accountant would grow
    roughly with sqrt(rounds) instead of linearly -- but it is simple,
    transparent, and always a safe (conservative) upper bound to report."""
    return rounds * per_round_epsilon(noise_multiplier, delta)


# ---------------------------------------------------------------------------
# Tighter accounting: Renyi-DP (RDP) accountant for the Gaussian mechanism
# ---------------------------------------------------------------------------
# Every round, ALL clients participate (no client sub-sampling), so there is
# no privacy-amplification-by-sampling -- each round is exactly one Gaussian
# mechanism with sensitivity = clip_norm and noise std = sigma * clip_norm.
# For that mechanism the RDP guarantee at order alpha is
#
#       eps_RDP(alpha) = alpha / (2 * sigma^2)            (Mironov, 2017)
#
# RDP composes ADDITIVELY over T rounds (T * alpha / (2 sigma^2)), and is
# converted to (eps, delta)-DP with the improved conversion of
# Canonne-Kamath-Steinke (2020) / Balle et al. (2020):
#
#       eps = eps_RDP(alpha) + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha-1)
#
# minimised over alpha. Growth in T is ~sqrt(T), much tighter than the basic
# linear composition above. Adjacency is client-level add/remove.
_ALPHAS = [1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256]


def rdp_epsilon(noise_multiplier, rounds, delta=1e-5):
    """(epsilon, delta)-DP bound after `rounds` Gaussian-mechanism releases
    with the given noise multiplier (client-level, full participation)."""
    if noise_multiplier is None or noise_multiplier <= 0:
        return float("inf")
    best = float("inf")
    for a in _ALPHAS:
        rdp = rounds * a / (2.0 * noise_multiplier ** 2)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)
