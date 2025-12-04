import xml.etree.ElementTree as ET
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Put your generated XML files in two folders, or name them consistently
# Example: "baseline_run_1.xml", "rl_run_1.xml"
BASELINE_PATTERN = "baseline_trip_*.xml"
RL_PATTERN = "rl_trip_*.xml"

def extract_waiting_times(file_pattern):
    """
    Parses all XML files matching the pattern and extracts
    the 'waitingTime' for every single vehicle.
    """
    files = glob.glob(file_pattern)
    all_wait_times = []
    
    if not files:
        print(f"⚠️ No files found for pattern: {file_pattern}")
        return np.array([])

    print(f"Loading {len(files)} files for pattern '{file_pattern}'...")
    
    for file in files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            # 'tripinfo' tag contains 'waitingTime' attribute
            # We explicitly cast to float
            waits = [float(trip.get('waitingTime')) for trip in root.findall('tripinfo')]
            all_wait_times.extend(waits)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    return np.array(all_wait_times)

def perform_analysis():
    # 1. Load Data
    print("--- Loading Data ---")
    baseline_waits = extract_waiting_times(BASELINE_PATTERN)
    rl_waits = extract_waiting_times(RL_PATTERN)

    if len(baseline_waits) == 0 or len(rl_waits) == 0:
        print("Error: Not enough data to compare. Make sure XML files exist.")
        return

    # 2. Descriptive Statistics
    print("\n--- Descriptive Statistics (Waiting Time in seconds) ---")
    print(f"{'Metric':<20} | {'Baseline':<15} | {'RL Model':<15}")
    print("-" * 56)
    print(f"{'Count (Vehicles)':<20} | {len(baseline_waits):<15} | {len(rl_waits):<15}")
    print(f"{'Mean':<20} | {np.mean(baseline_waits):.4f}          | {np.mean(rl_waits):.4f}")
    print(f"{'Median':<20} | {np.median(baseline_waits):.4f}          | {np.median(rl_waits):.4f}")
    print(f"{'Std Dev':<20} | {np.std(baseline_waits):.4f}          | {np.std(rl_waits):.4f}")
    print(f"{'Max Wait':<20} | {np.max(baseline_waits):.4f}          | {np.max(rl_waits):.4f}")

    # 3. Normality Test (Shapiro-Wilk)
    # Note: If N > 5000, Shapiro is often too sensitive. We use it here for formality,
    # but traffic data is almost never normal (it's usually Log-Normal).
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    
    # We take a random sample if data is huge to keep Shapiro valid
    sample_size = min(4000, len(baseline_waits), len(rl_waits))
    base_sample = np.random.choice(baseline_waits, sample_size, replace=False)
    rl_sample = np.random.choice(rl_waits, sample_size, replace=False)

    _, p_base = stats.shapiro(base_sample)
    _, p_rl = stats.shapiro(rl_sample)

    print(f"Baseline Normality p-value: {p_base:.5f} ({'Normal' if p_base > 0.05 else 'Not Normal'})")
    print(f"RL Model Normality p-value: {p_rl:.5f} ({'Normal' if p_rl > 0.05 else 'Not Normal'})")

    # 4. Hypothesis Testing
    print("\n--- Hypothesis Test ---")
    
    # If both are normal, use T-Test. Otherwise, Mann-Whitney U.
    if p_base > 0.05 and p_rl > 0.05:
        print("Distribution is Normal. Using Welch's T-Test.")
        stat, p_val = stats.ttest_ind(baseline_waits, rl_waits, equal_var=False)
        test_name = "Welch's T-Test"
    else:
        print("Distribution is Non-Normal. Using Mann-Whitney U Test.")
        # alternative='greater' means we test if Baseline > RL (i.e., RL reduced the time)
        stat, p_val = stats.mannwhitneyu(baseline_waits, rl_waits, alternative='greater')
        test_name = "Mann-Whitney U"

    print(f"Test: {test_name}")
    print(f"Statistic: {stat:.4f}")
    print(f"P-Value:   {p_val:.10f}")

    alpha = 0.05
    if p_val < alpha:
        print("\n✅ RESULT: SIGNIFICANT DIFFERENCE FOUND")
        print("The RL model performs significantly better (lower waiting times) than the baseline.")
    else:
        print("\n❌ RESULT: NO SIGNIFICANT DIFFERENCE")
        print("We cannot reject the null hypothesis. The improvements might be due to chance.")

    base_99 = np.percentile(baseline_waits, 100)
    rl_99 = np.percentile(rl_waits, 100)

    print(f"\n--- Fairness Check (99th Percentile) ---")
    print(f"Baseline 99% wait: {base_99:.2f}s")
    print(f"RL Model 99% wait: {rl_99:.2f}s")
    
    if rl_99 > base_99 * 1.5:
        print("⚠️ CRITICAL FAILURE: The RL Model is starving specific lanes (Unfair).")
        print("   The 99th percentile wait time is significantly worse.")
    elif rl_99 < base_99:
        print("✅ FAIRNESS IMPROVEMENT: The RL Model reduced extreme waiting times.")
    else:
        print("⚖️ FAIRNESS NEUTRAL.")

    # 5. Visualization
    print("\nGenerating Histogram...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(baseline_waits, fill=True, label='Baseline (Standard)', color='red', alpha=0.3)
    sns.kdeplot(rl_waits, fill=True, label='RL Agent', color='blue', alpha=0.3)
    plt.title('Distribution of Vehicle Waiting Times')
    plt.xlabel('Waiting Time (seconds)')
    plt.xlim(0, max(np.percentile(baseline_waits, 99), np.percentile(rl_waits, 99))) # Cut off extreme outliers for view
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_plot.png")
    print("Plot saved to 'comparison_plot.png'")

if __name__ == "__main__":
    perform_analysis()
