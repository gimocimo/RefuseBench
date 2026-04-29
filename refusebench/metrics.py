"""Statistical primitives for the benchmark.

- Wilson 95% CI on binomial proportions (every reported rate gets one)
- Cohen's kappa (2-rater agreement, used for judge-vs-human)
- Krippendorff's alpha for nominal data (k-rater agreement, used among LLM judges)
- Confusion matrix helper

All implementations are explicit (no scipy/sklearn dependency) so the math is auditable
by anyone reading the source. References:
- Wilson 1927; Brown, Cai & DasGupta 2001 ("Interval Estimation for a Binomial Proportion")
- Cohen 1960
- Krippendorff 2011 ("Computing Krippendorff's Alpha-Reliability")
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point * 100:.1f}% [{self.lo * 100:.1f}, {self.hi * 100:.1f}]"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> CI:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal-approximation interval at small n and at extreme
    proportions. z=1.96 → 95% CI; z=2.576 → 99% CI.
    """
    if n == 0:
        return CI(point=0.0, lo=0.0, hi=1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return CI(point=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half))


def cohens_kappa(
    rater_a: list[str], rater_b: list[str], categories: list[str] | None = None
) -> float:
    """Cohen's kappa: agreement between two raters on categorical data, chance-corrected.

    Range: -1 to 1. >= 0.6 substantial; >= 0.8 excellent.
    Returns 1.0 if both raters perfectly agree on every item.
    Returns 0.0 if there are no items to score.
    """
    assert len(rater_a) == len(rater_b), "rater lists must be the same length"
    n = len(rater_a)
    if n == 0:
        return 0.0
    cats = categories or sorted(set(rater_a) | set(rater_b))
    p_observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    a_counts = Counter(rater_a)
    b_counts = Counter(rater_b)
    p_expected = sum((a_counts[c] / n) * (b_counts[c] / n) for c in cats)
    if p_expected >= 1.0:
        # Both raters used a single category exclusively. Perfect agreement is uninformative;
        # by convention return 1.0 if observed agreement is also perfect, else 0.0.
        return 1.0 if p_observed >= 1.0 else 0.0
    return (p_observed - p_expected) / (1 - p_expected)


def krippendorff_alpha_nominal(
    reliability_data: list[list[str | None]], categories: list[str] | None = None
) -> float:
    """Krippendorff's alpha for nominal data, k raters, possibly missing values.

    `reliability_data[i][r]` is rater r's value for unit i, or None if missing.
    Each unit must have at least 2 non-None values to contribute.

    Returns alpha in roughly [-1, 1]. Conventional thresholds:
      >= 0.80 reliable | 0.67-0.80 tentative | < 0.67 unreliable
    """
    units: list[list[str]] = [
        [v for v in row if v is not None] for row in reliability_data
    ]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return 0.0

    cats = categories or sorted({v for u in units for v in u})

    n_total = sum(len(u) for u in units)
    if n_total < 2:
        return 0.0

    # Observed disagreement: for each unit, fraction of rater pairs that disagree, weighted by m-1.
    # D_o = sum over units of (1 / (m-1)) * sum over (c,k) c!=k of n_uc * n_uk, all divided by n_total.
    numerator_obs = 0.0
    for u in units:
        m = len(u)
        if m < 2:
            continue
        counts = Counter(u)
        # Disagreement contribution from this unit:
        # sum_{c != k} n_c * n_k = (sum n_c)^2 - sum n_c^2 = m^2 - sum n_c^2
        disagree_pairs = m * m - sum(c**2 for c in counts.values())
        numerator_obs += disagree_pairs / (m - 1)
    d_observed = numerator_obs / n_total

    # Expected disagreement: based on overall marginal distribution of categories.
    overall = Counter(v for u in units for v in u)
    # D_e = (1 / (n_total * (n_total - 1))) * sum_{c != k} n_c * n_k
    total_pairs_diff = n_total * n_total - sum(c**2 for c in overall.values())
    d_expected = total_pairs_diff / (n_total * (n_total - 1))

    if d_expected == 0.0:
        return 1.0 if d_observed == 0.0 else 0.0
    return 1.0 - d_observed / d_expected


def confusion_matrix(
    truth: list[str], predicted: list[str], categories: list[str] | None = None
) -> dict[tuple[str, str], int]:
    """Returns a dict keyed by (true_label, predicted_label) -> count."""
    assert len(truth) == len(predicted), "lists must be the same length"
    cats = categories or sorted(set(truth) | set(predicted))
    out: dict[tuple[str, str], int] = {(t, p): 0 for t in cats for p in cats}
    for t, p in zip(truth, predicted):
        out[(t, p)] += 1
    return out


def alpha_reliability_label(alpha: float) -> str:
    """Human-readable label for a Krippendorff's alpha value."""
    if alpha >= 0.80:
        return "reliable"
    if alpha >= 0.67:
        return "tentative"
    return "unreliable"
