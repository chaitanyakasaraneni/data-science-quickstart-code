"""
Chapter 5: Statistical Foundations of Data Science
Section 5.3 — Hypothesis Testing and Its Importance

Demonstrates key hypothesis testing concepts:
- Independent samples t-test
- Chi-square test of independence
- One-way ANOVA
- Interpreting p-values and confidence intervals
"""

import numpy as np
from scipy import stats


def independent_t_test():
    """Test whether a new website design improves conversion rates."""
    print("=" * 60)
    print("TEST 1: Independent Samples T-Test")
    print("=" * 60)
    print("Scenario: Does a new website design improve conversion rates?")
    print("  H₀: No difference between control and treatment groups")
    print("  H₁: Treatment group has a different conversion rate\n")

    np.random.seed(42)

    # Simulated daily conversion rates over 30 days
    control = np.random.normal(loc=0.12, scale=0.03, size=30)
    treatment = np.random.normal(loc=0.15, scale=0.03, size=30)

    t_stat, p_value = stats.ttest_ind(control, treatment)

    print(f"  Control group mean:   {control.mean():.4f} ({control.mean()*100:.2f}%)")
    print(f"  Treatment group mean: {treatment.mean():.4f} ({treatment.mean()*100:.2f}%)")
    print(f"  T-statistic:          {t_stat:.4f}")
    print(f"  P-value:              {p_value:.6f}")
    print()

    alpha = 0.05
    if p_value < alpha:
        print(f"  ✓ Result: SIGNIFICANT (p={p_value:.6f} < α={alpha})")
        print("  → Reject H₀. The new design shows a statistically significant effect.")
    else:
        print(f"  ✗ Result: NOT significant (p={p_value:.4f} >= α={alpha})")
        print("  → Fail to reject H₀. No significant difference detected.")

    # Confidence interval for the difference in means
    diff = treatment.mean() - control.mean()
    se = np.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    print(f"\n  95% CI for difference in means: [{ci_lower:.4f}, {ci_upper:.4f}]")


def chi_square_test():
    """Test whether product preference is independent of customer region."""
    print("\n\n" + "=" * 60)
    print("TEST 2: Chi-Square Test of Independence")
    print("=" * 60)
    print("Scenario: Is product preference independent of customer region?")
    print("  H₀: Product preference and region are independent")
    print("  H₁: Product preference depends on region\n")

    # Observed frequencies: rows = regions, columns = product preferences
    # Regions: North, South, East, West
    # Products: Electronics, Clothing, Home
    observed = np.array([
        [120, 80, 50],   # North
        [90, 110, 60],   # South
        [100, 95, 55],   # East
        [85, 70, 95],    # West
    ])

    regions = ["North", "South", "East", "West"]
    products = ["Electronics", "Clothing", "Home"]

    print("  Observed frequencies:")
    print(f"  {'Region':<10} {'Electronics':>12} {'Clothing':>10} {'Home':>8}")
    print("  " + "-" * 42)
    for i, region in enumerate(regions):
        print(f"  {region:<10} {observed[i, 0]:>12} {observed[i, 1]:>10} {observed[i, 2]:>8}")

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    print(f"\n  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom:  {dof}")
    print(f"  P-value:             {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n  ✓ Result: SIGNIFICANT (p={p_value:.6f} < α={alpha})")
        print("  → Reject H₀. Product preference depends on region.")
    else:
        print(f"\n  ✗ Result: NOT significant (p={p_value:.4f} >= α={alpha})")
        print("  → Fail to reject H₀. No evidence of dependence.")


def one_way_anova():
    """Test whether mean satisfaction scores differ across service tiers."""
    print("\n\n" + "=" * 60)
    print("TEST 3: One-Way ANOVA")
    print("=" * 60)
    print("Scenario: Do customer satisfaction scores differ across service tiers?")
    print("  H₀: All group means are equal")
    print("  H₁: At least one group mean differs\n")

    np.random.seed(42)

    basic = np.random.normal(loc=6.5, scale=1.2, size=50)
    standard = np.random.normal(loc=7.2, scale=1.0, size=50)
    premium = np.random.normal(loc=8.1, scale=0.9, size=50)

    groups = {"Basic": basic, "Standard": standard, "Premium": premium}

    for name, data in groups.items():
        print(f"  {name:<10} mean={data.mean():.2f}  std={data.std():.2f}  n={len(data)}")

    f_stat, p_value = stats.f_oneway(basic, standard, premium)

    print(f"\n  F-statistic: {f_stat:.4f}")
    print(f"  P-value:     {p_value:.8f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n  ✓ Result: SIGNIFICANT (p={p_value:.8f} < α={alpha})")
        print("  → Reject H₀. Satisfaction scores differ significantly across tiers.")
        print("\n  Post-hoc pairwise comparisons (Bonferroni-corrected α = 0.017):")
        pairs = [("Basic", "Standard"), ("Basic", "Premium"), ("Standard", "Premium")]
        for g1_name, g2_name in pairs:
            _, p = stats.ttest_ind(groups[g1_name], groups[g2_name])
            sig = "✓ Significant" if p < 0.017 else "✗ Not significant"
            print(f"    {g1_name} vs {g2_name}: p={p:.6f} → {sig}")
    else:
        print(f"\n  ✗ Result: NOT significant (p={p_value:.4f} >= α={alpha})")


def effect_size_demo():
    """Demonstrate why p-value alone is insufficient — effect size matters."""
    print("\n\n" + "=" * 60)
    print("BONUS: Why Effect Size Matters")
    print("=" * 60)
    print("Two experiments with the same effect but different sample sizes:\n")

    np.random.seed(42)

    # Small sample — same effect, not significant
    small_ctrl = np.random.normal(50, 10, 15)
    small_treat = np.random.normal(55, 10, 15)
    _, p_small = stats.ttest_ind(small_ctrl, small_treat)

    # Large sample — same effect, significant
    large_ctrl = np.random.normal(50, 10, 500)
    large_treat = np.random.normal(55, 10, 500)
    _, p_large = stats.ttest_ind(large_ctrl, large_treat)

    # Cohen's d
    def cohens_d(g1, g2):
        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        return (g2.mean() - g1.mean()) / pooled_std

    d_small = cohens_d(small_ctrl, small_treat)
    d_large = cohens_d(large_ctrl, large_treat)

    print(f"  Small sample (n=15):  p={p_small:.4f}  Cohen's d={d_small:.3f}")
    print(f"  Large sample (n=500): p={p_large:.8f}  Cohen's d={d_large:.3f}")
    print(f"\n  Both have similar effect sizes (~0.5, medium), but only the large")
    print(f"  sample reaches statistical significance. Always report effect size")
    print(f"  alongside p-values to give the full picture.")


if __name__ == "__main__":
    independent_t_test()
    chi_square_test()
    one_way_anova()
    effect_size_demo()
