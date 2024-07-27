import numpy as np
from scipy import stats

# Data
baseline_scans = [
    1.658022072, 2.000783331, 1.401217321, 1.638375936, 1.29506031,
    1.546389665, 1.418182294, 1.648848093, 3.088362835, 1.956389551,
    1.350675967, 1.576799401, 1.460127227, 1.613973927
]

interim_scans = [
    1.664671514, 1.953374511, 1.397220726, 1.676305607, 1.361504467,
    1.587508992, 1.282413291, 1.584525408, 1.553085908, 1.6995245,
    0.8935968976, 1.128503777, 1.322722766, 1.514105146
]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(baseline_scans, interim_scans)

# Output the result
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Determine if the difference is significant
alpha = 0.05  # Commonly used significance level
if p_value < alpha:
    print("The differences are statistically significant.")
else:
    print("The differences are not statistically significant.")