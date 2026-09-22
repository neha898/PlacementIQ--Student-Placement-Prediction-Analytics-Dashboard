"""
main.py
=======
PlacementIQ — Student Career Success Analysis
Data Analytics Project

This is the single entry-point file for the entire project.
It contains two modes:

  1. ANALYSIS MODE  — runs data cleaning, metrics, charts (terminal output)
     Usage: python main.py

  2. DASHBOARD MODE — launches the interactive Streamlit dashboard
     Usage: python -m streamlit run main.py

When run directly (python main.py), the analysis pipeline executes.
When run via Streamlit, the dashboard renders automatically.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import sys
import io
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_PATH = "student_career_success_dataset_cleaned.csv"
CHARTS_DIR   = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Detect if we are running inside Streamlit or as a plain script
STREAMLIT_MODE = "streamlit" in sys.modules or "streamlit.runtime" in sys.modules

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  PART A — DATA CLEANING & ANALYSIS
#  (runs when: python main.py)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts and servers
import matplotlib.pyplot as plt
import seaborn as sns

# Chart colour palette for matplotlib (analysis charts)
PALETTE_MAIN = "#3b82d4"
PALETTE_PAIR = ["#3b82d4", "#f87171"]
sns.set_theme(style="whitegrid", palette="muted")


# ── STEP 1: Load and validate ────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load the CSV and print a structural summary:
      - Shape (rows x columns)
      - Column data types
      - Null value counts per column
    Returns the raw DataFrame.
    """
    df = pd.read_csv(path)

    print("=" * 60)
    print("STEP 1 — LOAD AND VALIDATE DATASET STRUCTURE")
    print("=" * 60)
    print(f"Shape        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")

    null_counts = df.isnull().sum()
    print(f"\nNull values per column:\n{null_counts[null_counts >= 0].to_string()}")
    print(f"\nTotal null cells : {null_counts.sum()}")

    return df


# ── STEP 2: Data cleaning checks ─────────────────────────────────────────────

def run_cleaning_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive data-quality checks:
      2a. Missing values
      2b. Duplicate rows
      2c. Duplicate Student_IDs
      2d. Categorical consistency (expected value sets per column)
      2e. Numeric range validation
      2f. Logical consistency (Placement_Status vs salary/tier/mode)
      2g. Outlier detection and capping using the IQR method
    Returns the cleaned DataFrame.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — DATA CLEANING CHECKS")
    print("=" * 60)

    # 2a. Missing values
    print(f"\n[2a] Total missing values : {df.isnull().sum().sum()}")

    # 2b. Duplicate rows
    print(f"[2b] Duplicate rows       : {df.duplicated().sum()}")

    # 2c. Duplicate Student_IDs
    print(f"[2c] Duplicate Student_IDs: {df['Student_ID'].duplicated().sum()}")

    # 2d. Categorical consistency — expected value sets
    expected_cats = {
        "Placement_Status"     : {"Placed", "Not Placed"},
        "Company_Tier"         : {"Tier 1", "Tier 2", "Tier 3", "No Company"},
        "Placement_Mode"       : {"Campus Placement", "Off-Campus",
                                  "Internship Conversion", "Referral", "Not Applicable"},
        "Gender"               : {"Male", "Female", "Other"},
        "University_Year"      : {"Freshman", "Sophomore", "Junior", "Senior"},
        "GitHub_Profile"       : {"Yes", "No"},
        "LinkedIn_Profile"     : {"Yes", "No"},
        "Leadership_Experience": {"Yes", "No"},
    }
    print("\n[2d] Categorical consistency check:")
    for col, valid_vals in expected_cats.items():
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - valid_vals
        print(f"     {col:25s} — unexpected values: {unexpected if unexpected else 'None'}")

    # 2e. Numeric range validation
    numeric_ranges = {
        "Age"                  : (17, 30),
        "Attendance_Percentage": (0, 100),
        "Study_Hours_Per_Week" : (0, 168),
        "CGPA"                 : (0.0, 4.0),
        "Programming_Skill"    : (1, 10),
        "Projects_Completed"   : (0, 50),
        "Certifications"       : (0, 20),
        "Hackathons"           : (0, 20),
        "Internships"          : (0, 10),
        "Resume_Score"         : (0, 100),
        "Communication_Skills" : (1, 10),
        "Teamwork"             : (1, 10),
        "Problem_Solving"      : (1, 10),
        "Interview_Score"      : (0, 100),
        "Employability_Score"  : (0, 500),
        "Starting_Salary_USD"  : (0, 300_000),
    }
    print("\n[2e] Numeric range validation:")
    for col, (lo, hi) in numeric_ranges.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"     {col:30s} [{lo}, {hi}] — out-of-range rows: {out_of_range}")

    # 2f. Logical consistency between Placement_Status and salary / tier / mode
    print("\n[2f] Logical consistency check (Placement_Status vs salary/tier/mode):")
    not_placed = df[df["Placement_Status"] == "Not Placed"]
    placed     = df[df["Placement_Status"] == "Placed"]
    print(f"     Not Placed with salary != 0        : {(not_placed['Starting_Salary_USD'] != 0).sum()}")
    print(f"     Not Placed with tier != No Company : {(not_placed['Company_Tier'] != 'No Company').sum()}")
    print(f"     Not Placed with mode != Not Applicable: {(not_placed['Placement_Mode'] != 'Not Applicable').sum()}")
    print(f"     Placed with salary = 0             : {(placed['Starting_Salary_USD'] == 0).sum()}")

    # 2g. Outlier detection — IQR method — cap (not drop)
    print("\n[2g] Outlier detection (IQR method — cap to whisker boundaries):")
    for col in ["Study_Hours_Per_Week", "CGPA"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"     {col:30s} — outliers: {outliers}  (range [{lower:.2f}, {upper:.2f}])")
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Salary outliers — checked only among Placed students (Not Placed always = 0)
    sal_col  = "Starting_Salary_USD"
    df_p     = df[df["Placement_Status"] == "Placed"]
    q1s, q3s = df_p[sal_col].quantile(0.25), df_p[sal_col].quantile(0.75)
    iqrs     = q3s - q1s
    lower_s, upper_s = q1s - 1.5 * iqrs, q3s + 1.5 * iqrs
    sal_out  = ((df_p[sal_col] < lower_s) | (df_p[sal_col] > upper_s)).sum()
    print(f"     {'Starting_Salary_USD (Placed only)':30s} — outliers: {sal_out}  "
          f"(range [{lower_s:,.0f}, {upper_s:,.0f}])")
    df.loc[df["Placement_Status"] == "Placed", sal_col] = \
        df.loc[df["Placement_Status"] == "Placed", sal_col].clip(lower=lower_s, upper=upper_s)

    print("\n[Cleaning] Dataset ready for analysis.\n")
    return df


# ── STEP 3: Key metrics ───────────────────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate all key metrics:
      3.1  Overall placement rate
      3.2  Placement rate by Major and University Year
      3.3  Avg CGPA and Employability Score: Placed vs Not Placed
      3.4  Avg starting salary by Company Tier
      3.5  Internships / Certifications / GitHub vs placement rate
      3.6  Top career fields (placed students only)
      3.7  Avg salary by Major (placed students only)
    Returns a dictionary of result DataFrames and scalar values.
    """
    print("=" * 60)
    print("STEP 3 — KEY METRICS CALCULATION")
    print("=" * 60)

    results     = {}
    placed_mask = df["Placement_Status"] == "Placed"
    placed_df   = df[placed_mask].copy()

    # 3.1 Overall placement rate
    overall_rate = placed_mask.mean() * 100
    results["overall_placement_rate"] = round(overall_rate, 2)
    print(f"\n[3.1] Overall placement rate : {overall_rate:.2f}%")

    # 3.2a Placement rate by Major
    major_rate = (df.groupby("Major")["Placement_Status"]
                  .apply(lambda s: (s == "Placed").mean() * 100)
                  .reset_index(name="Placement_Rate_%")
                  .sort_values("Placement_Rate_%", ascending=False))
    results["placement_by_major"] = major_rate
    print(f"\n[3.2a] Placement rate by Major:\n{major_rate.to_string(index=False)}")

    # 3.2b Placement rate by University Year
    year_order = ["Freshman", "Sophomore", "Junior", "Senior"]
    year_rate  = (df.groupby("University_Year")["Placement_Status"]
                  .apply(lambda s: (s == "Placed").mean() * 100)
                  .reset_index(name="Placement_Rate_%"))
    year_rate["University_Year"] = pd.Categorical(
        year_rate["University_Year"], categories=year_order, ordered=True)
    year_rate = year_rate.sort_values("University_Year")
    results["placement_by_year"] = year_rate
    print(f"\n[3.2b] Placement rate by University_Year:\n{year_rate.to_string(index=False)}")

    # 3.3 Avg CGPA and Employability Score: Placed vs Not Placed
    comparison = (df.groupby("Placement_Status")[["CGPA", "Employability_Score"]]
                  .mean().round(2).reset_index())
    results["placed_vs_not"] = comparison
    print(f"\n[3.3] Avg CGPA & Employability Score — Placed vs Not Placed:\n"
          f"{comparison.to_string(index=False)}")

    # 3.4 Avg starting salary by Company Tier
    salary_by_tier = (placed_df.groupby("Company_Tier")["Starting_Salary_USD"]
                      .mean().round(0)
                      .reset_index(name="Avg_Starting_Salary_USD")
                      .sort_values("Avg_Starting_Salary_USD", ascending=False))
    results["salary_by_tier"] = salary_by_tier
    print(f"\n[3.4] Avg Starting Salary by Company Tier:\n{salary_by_tier.to_string(index=False)}")

    # 3.5a Placement rate by Internship count
    internship_rate = (df.groupby("Internships")["Placement_Status"]
                       .apply(lambda s: (s == "Placed").mean() * 100)
                       .reset_index(name="Placement_Rate_%").sort_values("Internships"))
    results["placement_by_internships"] = internship_rate
    print(f"\n[3.5a] Placement rate by Internship count:\n{internship_rate.to_string(index=False)}")

    # 3.5b Placement rate by Certification count
    cert_rate = (df.groupby("Certifications")["Placement_Status"]
                 .apply(lambda s: (s == "Placed").mean() * 100)
                 .reset_index(name="Placement_Rate_%").sort_values("Certifications"))
    results["placement_by_certs"] = cert_rate
    print(f"\n[3.5b] Placement rate by Certification count:\n{cert_rate.to_string(index=False)}")

    # 3.5c Placement rate by GitHub Profile
    github_rate = (df.groupby("GitHub_Profile")["Placement_Status"]
                   .apply(lambda s: (s == "Placed").mean() * 100)
                   .reset_index(name="Placement_Rate_%"))
    results["placement_by_github"] = github_rate
    print(f"\n[3.5c] Placement rate by GitHub Profile:\n{github_rate.to_string(index=False)}")

    # 3.6 Top 10 Career Fields
    top_fields = (placed_df["Career_Field"].value_counts()
                  .reset_index().rename(columns={"count": "Count"}).head(10))
    results["top_career_fields"] = top_fields
    print(f"\n[3.6] Top 10 Career Fields:\n{top_fields.to_string(index=False)}")

    # 3.7 Avg salary by Major (placed students only)
    salary_by_major = (placed_df.groupby("Major")["Starting_Salary_USD"]
                       .mean().round(0)
                       .reset_index(name="Avg_Starting_Salary_USD")
                       .sort_values("Avg_Starting_Salary_USD", ascending=False))
    results["salary_by_major"] = salary_by_major
    print(f"\n[3.7] Avg Starting Salary by Major:\n{salary_by_major.to_string(index=False)}")

    return results


# ── STEP 4: Group summaries ───────────────────────────────────────────────────

def group_summaries(df: pd.DataFrame) -> None:
    """
    Print grouped summary tables with totals, counts, percentages, and averages:
      4a. Placement status breakdown
      4b. Placement mode breakdown (placed only)
      4c. Company tier breakdown (placed only)
      4d. Avg CGPA / Employability / Interview Score by Major
    """
    print("\n" + "=" * 60)
    print("STEP 4 — GROUP AND SUMMARIZE DATA")
    print("=" * 60)

    # 4a. Placement status breakdown
    sc = df["Placement_Status"].value_counts().reset_index()
    sc.columns = ["Status", "Count"]
    sc["Percentage_%"] = (sc["Count"] / len(df) * 100).round(2)
    print(f"\n[4a] Placement Status Breakdown:\n{sc.to_string(index=False)}")

    # 4b. Placement mode breakdown (placed only)
    mc = df[df["Placement_Status"] == "Placed"]["Placement_Mode"].value_counts().reset_index()
    mc.columns = ["Placement_Mode", "Count"]
    mc["Percentage_%"] = (mc["Count"] / mc["Count"].sum() * 100).round(2)
    print(f"\n[4b] Placement Mode Breakdown (placed only):\n{mc.to_string(index=False)}")

    # 4c. Company tier breakdown (placed only)
    tc = df[df["Placement_Status"] == "Placed"]["Company_Tier"].value_counts().reset_index()
    tc.columns = ["Company_Tier", "Count"]
    tc["Percentage_%"] = (tc["Count"] / tc["Count"].sum() * 100).round(2)
    print(f"\n[4c] Company Tier Breakdown (placed only):\n{tc.to_string(index=False)}")

    # 4d. Avg scores by Major
    avg_by_major = (df.groupby("Major")[["CGPA", "Employability_Score", "Interview_Score"]]
                    .mean().round(2).reset_index()
                    .sort_values("Employability_Score", ascending=False))
    print(f"\n[4d] Avg CGPA / Employability / Interview Score by Major:\n"
          f"{avg_by_major.to_string(index=False)}")


# ── STEP 5: Chart generation ──────────────────────────────────────────────────

def save_chart(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure as PNG to the charts directory."""
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def create_charts(df: pd.DataFrame, results: dict) -> None:
    """
    Generate and save 12 analysis charts to the charts/ folder:
      1.  Placement rate by Major (horizontal bar)
      2.  Placement rate by University Year (bar)
      3.  Avg CGPA: Placed vs Not Placed (bar)
      4.  Avg Employability Score: Placed vs Not Placed (bar)
      5.  Avg Starting Salary by Company Tier (bar)
      6.  Placement rate by Internship count (line + bar combo)
      7.  Placement rate by GitHub Profile (bar)
      8.  Placement Status distribution (pie)
      9.  Placement Mode distribution (pie)
      10. CGPA distribution: Placed vs Not Placed (boxplot)
      11. Employability Score distribution: Placed vs Not Placed (boxplot)
      12. Correlation heatmap of key numeric features
    """
    print("\n" + "=" * 60)
    print("STEP 5 — CREATING CHARTS")
    print("=" * 60)

    # Chart 1: Placement rate by Major
    fig, ax = plt.subplots(figsize=(9, 5))
    data = results["placement_by_major"].sort_values("Placement_Rate_%")
    ax.barh(data["Major"], data["Placement_Rate_%"], color=PALETTE_MAIN)
    ax.set_xlabel("Placement Rate (%)")
    ax.set_title("Placement Rate by Major")
    ax.set_xlim(0, 100)
    for bar, val in zip(ax.patches, data["Placement_Rate_%"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    save_chart(fig, "placement_by_major.png")

    # Chart 2: Placement rate by University Year
    fig, ax = plt.subplots(figsize=(7, 4))
    data = results["placement_by_year"]
    bars = ax.bar(data["University_Year"], data["Placement_Rate_%"], color=PALETTE_MAIN)
    ax.set_ylabel("Placement Rate (%)")
    ax.set_title("Placement Rate by University Year")
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    save_chart(fig, "placement_by_year.png")

    # Chart 3: Avg CGPA — Placed vs Not Placed
    fig, ax = plt.subplots(figsize=(6, 4))
    comp = results["placed_vs_not"]
    bars = ax.bar(comp["Placement_Status"], comp["CGPA"], color=PALETTE_PAIR)
    ax.set_ylabel("Average CGPA")
    ax.set_title("Average CGPA: Placed vs Not Placed")
    ax.set_ylim(0, 4.0)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_chart(fig, "cgpa_placed_vs_not.png")

    # Chart 4: Avg Employability Score — Placed vs Not Placed
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(comp["Placement_Status"], comp["Employability_Score"], color=PALETTE_PAIR)
    ax.set_ylabel("Average Employability Score")
    ax.set_title("Average Employability Score: Placed vs Not Placed")
    ax.set_ylim(0, comp["Employability_Score"].max() * 1.15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_chart(fig, "emp_score_placed_vs_not.png")

    # Chart 5: Avg Starting Salary by Company Tier
    fig, ax = plt.subplots(figsize=(7, 4))
    data = results["salary_by_tier"]
    bars = ax.bar(data["Company_Tier"], data["Avg_Starting_Salary_USD"], color=PALETTE_MAIN)
    ax.set_ylabel("Avg Starting Salary (USD)")
    ax.set_title("Average Starting Salary by Company Tier")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"${bar.get_height():,.0f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    save_chart(fig, "salary_by_tier.png")

    # Chart 6: Placement rate by Internship count (line + bar combo)
    fig, ax = plt.subplots(figsize=(8, 4))
    data = results["placement_by_internships"]
    ax.bar(data["Internships"].astype(str), data["Placement_Rate_%"],
           color=PALETTE_MAIN, alpha=0.7)
    ax.plot(data["Internships"].astype(str), data["Placement_Rate_%"],
            marker="o", color="#f87171", linewidth=2)
    ax.set_xlabel("Number of Internships")
    ax.set_ylabel("Placement Rate (%)")
    ax.set_title("Placement Rate by Number of Internships")
    ax.set_ylim(0, 110)
    for x, y in zip(data["Internships"].astype(str), data["Placement_Rate_%"]):
        ax.text(x, y + 2, f"{y:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    save_chart(fig, "placement_by_internships.png")

    # Chart 7: Placement rate by GitHub Profile
    fig, ax = plt.subplots(figsize=(5, 4))
    data = results["placement_by_github"]
    bars = ax.bar(data["GitHub_Profile"], data["Placement_Rate_%"], color=PALETTE_PAIR)
    ax.set_ylabel("Placement Rate (%)")
    ax.set_title("Placement Rate: GitHub Profile (Yes vs No)")
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_chart(fig, "placement_by_github.png")

    # Chart 8: Placement Status pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df["Placement_Status"].value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=PALETTE_PAIR, startangle=140)
    ax.set_title("Placement Status Distribution")
    plt.tight_layout()
    save_chart(fig, "placement_status_pie.png")

    # Chart 9: Placement Mode pie chart (placed only)
    fig, ax = plt.subplots(figsize=(6, 5))
    placed_df = df[df["Placement_Status"] == "Placed"]
    mode_counts = placed_df["Placement_Mode"].value_counts()
    ax.pie(mode_counts, labels=mode_counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Placement Mode Distribution (Placed Students)")
    plt.tight_layout()
    save_chart(fig, "placement_mode_pie.png")

    # Chart 10: CGPA boxplot — Placed vs Not Placed
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Placement_Status", y="CGPA", hue="Placement_Status",
                palette={"Placed": PALETTE_PAIR[0], "Not Placed": PALETTE_PAIR[1]},
                legend=False, ax=ax)
    ax.set_title("CGPA Distribution: Placed vs Not Placed")
    plt.tight_layout()
    save_chart(fig, "cgpa_boxplot.png")

    # Chart 11: Employability Score boxplot — Placed vs Not Placed
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Placement_Status", y="Employability_Score", hue="Placement_Status",
                palette={"Placed": PALETTE_PAIR[0], "Not Placed": PALETTE_PAIR[1]},
                legend=False, ax=ax)
    ax.set_title("Employability Score Distribution: Placed vs Not Placed")
    plt.tight_layout()
    save_chart(fig, "emp_score_boxplot.png")

    # Chart 12: Correlation heatmap
    numeric_cols = ["CGPA", "Study_Hours_Per_Week", "Internships", "Certifications",
                    "Programming_Skill", "Projects_Completed", "Resume_Score",
                    "Communication_Skills", "Interview_Score", "Employability_Score",
                    "Starting_Salary_USD"]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation Heatmap of Key Numeric Features")
    plt.tight_layout()
    save_chart(fig, "correlation_heatmap.png")

    print("\nAll charts saved successfully.")


# ── STEP 6: Recommendations ───────────────────────────────────────────────────

def print_recommendations() -> None:
    """Print 5 career-readiness and placement-strategy recommendations."""
    print("\n" + "=" * 60)
    print("STEP 6 — CAREER READINESS & PLACEMENT STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    recs = [
        "1. Internship support is the single highest-impact intervention: placement rate "
        "grows from ~7% (0 internships) to ~92% (5 internships). Placement cells should "
        "actively facilitate internship placements, especially for first- and second-year students.",
        "2. Students in lower-placement majors (Business Analytics, Electrical Engineering) "
        "should receive targeted upskilling — guided certification tracks and project mentorship.",
        "3. GitHub and LinkedIn profile creation should be encouraged from Year 1. "
        "Students with a GitHub profile have an 80% placement rate vs 69% without one.",
        "4. Career counselors should emphasise Tier 1 recruitment pathways: average salary "
        "at Tier 1 ($90,011) is nearly double that of Tier 3 ($46,958).",
        "5. Artificial Intelligence and Software Engineering graduates earn the highest salaries "
        "and have the best placement rates — highlight these programmes in career orientation sessions.",
    ]
    for rec in recs:
        print(f"\n  {rec}")


# ── Analysis pipeline entry point ─────────────────────────────────────────────

def run_analysis():
    """Run the full 6-step analysis pipeline (terminal mode)."""
    # Force UTF-8 output on Windows so unicode characters print correctly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    df      = load_and_validate(DATASET_PATH)
    df      = run_cleaning_checks(df)
    results = calculate_metrics(df)
    group_summaries(df)
    create_charts(df, results)
    print_recommendations()
    print("\nAnalysis complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  PART B — STREAMLIT DASHBOARD
#  (runs when: python -m streamlit run main.py)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

import plotly.express as px
import plotly.graph_objects as go


# ── Dashboard colour theme ────────────────────────────────────────────────────

BG   = "#1e293b"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTE = "#94a3b8"
C1   = "#3b82f6"   # blue  — Placed / primary
C2   = "#f87171"   # red   — Not Placed
CV   = "#7c3aed"   # violet — salary by major

def dark_layout(height=380, **extra):
    """Return Plotly layout kwargs for the dark theme."""
    base = dict(
        height=height, autosize=False,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, sans-serif", size=12),
        title_font=dict(color="#f1f5f9", size=14),
        xaxis=dict(gridcolor=GRID, linecolor="#475569",
                   tickfont=dict(color=MUTE), title_font=dict(color=MUTE)),
        yaxis=dict(gridcolor=GRID, linecolor="#475569",
                   tickfont=dict(color=MUTE), title_font=dict(color=MUTE)),
        legend=dict(bgcolor=BG, bordercolor=GRID, font=dict(color=TEXT)),
        margin=dict(l=10, r=20, t=45, b=20),
    )
    base.update(extra)
    return base

def chart_png(fig, width=680, height=380):
    """Render a Plotly figure as a static PNG (never grows in browser)."""
    return fig.to_image(format="png", width=width, height=height, scale=1.6)


# ── Data loader (cached so filters don't reload the file) ─────────────────────

def _load_data_for_dashboard(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Study_Hours_Per_Week", "CGPA"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    placed_mask = df["Placement_Status"] == "Placed"
    sal = df.loc[placed_mask, "Starting_Salary_USD"]
    q1s, q3s = sal.quantile(0.25), sal.quantile(0.75)
    iqrs = q3s - q1s
    df.loc[placed_mask, "Starting_Salary_USD"] = sal.clip(
        lower=q1s - 1.5 * iqrs, upper=q3s + 1.5 * iqrs)
    return df


# ── Dashboard entry point ─────────────────────────────────────────────────────

def run_dashboard():
    """Build and render the full Streamlit dashboard."""
    import streamlit as st

    # ── Page config
    st.set_page_config(
        page_title="PlacementIQ — Student Career Success Dashboard",
        page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

    # ── CSS
    st.markdown("""
    <style>
        .stApp, .block-container { background-color: #0f172a; }
        .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
        html, body, [class*="css"], .stMarkdown, p, span, label, div {
            color: #e2e8f0 !important;
            font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
        }
        h1 { color: #f1f5f9 !important; font-size: 2rem !important; font-weight: 800 !important; }
        .kpi-card {
            background: linear-gradient(135deg, #1e293b 0%, #1a2540 100%);
            border: 1px solid #334155; border-top: 3px solid #3b82f6;
            border-radius: 12px; padding: 20px 22px; text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }
        .kpi-label { font-size: 12px !important; color: #94a3b8 !important; margin-bottom: 6px;
                     font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; }
        .kpi-value { font-size: 30px !important; font-weight: 800 !important;
                     color: #f1f5f9 !important; line-height: 1.1; }
        .kpi-sub   { font-size: 11px !important; color: #64748b !important; margin-top: 4px; }
        .section-header { font-size: 17px; font-weight: 700; color: #f1f5f9 !important;
                          border-bottom: 2px solid #1e40af; padding-bottom: 8px; margin-bottom: 6px; }
        section[data-testid="stSidebar"] { background-color: #0f172a !important;
                                           border-right: 1px solid #1e293b; }
        section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
        .stMultiSelect [data-baseweb="tag"] { background-color: #1e40af !important; color: #e0e7ff !important; }
        hr { border-color: #1e293b !important; }
        .stDownloadButton > button { background-color: #1e40af !important; color: #e0e7ff !important;
                                     border: none !important; border-radius: 8px !important;
                                     font-weight: 600; padding: 10px 20px; }
        .stDownloadButton > button:hover { background-color: #2563eb !important; }
        .stImage img { max-width: 100% !important; height: auto !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Load data (cached)
    load_cached = st.cache_data(show_spinner="Loading dataset…")(_load_data_for_dashboard)
    df_raw = load_cached(DATASET_PATH)

    # ── Sidebar filters
    st.sidebar.title("🎓 PlacementIQ")
    st.sidebar.markdown("**Student Career Success Dashboard**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    year_order       = ["Freshman", "Sophomore", "Junior", "Senior"]
    selected_majors  = st.sidebar.multiselect("Major",           sorted(df_raw["Major"].unique()),  default=sorted(df_raw["Major"].unique()))
    selected_genders = st.sidebar.multiselect("Gender",          sorted(df_raw["Gender"].unique()), default=sorted(df_raw["Gender"].unique()))
    selected_years   = st.sidebar.multiselect("University Year", year_order,                        default=year_order)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<small style='color:#57606a'>Dataset: 50,000 student records<br>"
                        "Source: student_career_success_dataset_cleaned.csv</small>",
                        unsafe_allow_html=True)

    df = df_raw[df_raw["Major"].isin(selected_majors) &
                df_raw["Gender"].isin(selected_genders) &
                df_raw["University_Year"].isin(selected_years)].copy()

    if df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    placed_df = df[df["Placement_Status"] == "Placed"]

    # ── Page title
    st.title("🎓 PlacementIQ — Student Career Success Dashboard")
    st.markdown("Identifying the key drivers of student placement success using academic, "
                "skill, and experience data to enable data-driven career decision-making.")
    st.markdown("---")

    # ── KPI cards
    total_students = len(df)
    placement_rate = round((df["Placement_Status"] == "Placed").mean() * 100, 2)
    avg_salary     = round(placed_df["Starting_Salary_USD"].mean(), 0) if len(placed_df) > 0 else 0
    avg_emp_score  = round(df["Employability_Score"].mean(), 1)

    c1_, c2_, c3_, c4_ = st.columns(4)
    for col, label, value, sub in [
        (c1_, "Total Students",          f"{total_students:,}",  "in current filter"),
        (c2_, "Placement Rate",          f"{placement_rate}%",   "of filtered students placed"),
        (c3_, "Avg Starting Salary",     f"${avg_salary:,.0f}",  "placed students only"),
        (c4_, "Avg Employability Score", f"{avg_emp_score}",     "all filtered students"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{value}</div>'
                     f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Helper: render Plotly fig as static PNG image
    def show(fig, w=680, h=380):
        st.image(chart_png(fig, width=w, height=h), use_container_width=True)

    # ── Section 1: Placement Rates
    st.markdown('<div class="section-header">📊 Placement Rates</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        data = (df.groupby("Major")["Placement_Status"]
                .apply(lambda s: round((s == "Placed").mean() * 100, 2))
                .reset_index(name="Placement Rate (%)").sort_values("Placement Rate (%)"))
        fig = px.bar(data, x="Placement Rate (%)", y="Major", orientation="h",
                     text="Placement Rate (%)", color_discrete_sequence=[C1],
                     title="Placement Rate by Major")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(380, xaxis_range=[0, 108], xaxis_title="Placement Rate (%)", yaxis_title=""))
        show(fig)
    with col_b:
        data = (df.groupby("University_Year")["Placement_Status"]
                .apply(lambda s: round((s == "Placed").mean() * 100, 2))
                .reset_index(name="Placement Rate (%)"))
        data["University_Year"] = pd.Categorical(data["University_Year"], categories=year_order, ordered=True)
        data = data.sort_values("University_Year")
        fig = px.bar(data, x="University_Year", y="Placement Rate (%)",
                     text="Placement Rate (%)", color_discrete_sequence=[C1],
                     title="Placement Rate by University Year")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(380, yaxis_range=[0, 108], xaxis_title="", yaxis_title="Placement Rate (%)"))
        show(fig)

    # ── Section 2: Academic & Employability Comparison
    st.markdown('<div class="section-header">📚 Academic & Employability Comparison</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    comp = df.groupby("Placement_Status")[["CGPA", "Employability_Score"]].mean().round(2).reset_index()
    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.bar(comp, x="Placement_Status", y="CGPA", text="CGPA", color="Placement_Status",
                     color_discrete_map={"Placed": C1, "Not Placed": C2}, title="Average CGPA: Placed vs Not Placed")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(360, yaxis_range=[0, 4.3], xaxis_title="", yaxis_title="Average CGPA", showlegend=False))
        show(fig)
    with col_d:
        fig = px.bar(comp, x="Placement_Status", y="Employability_Score", text="Employability_Score",
                     color="Placement_Status", color_discrete_map={"Placed": C1, "Not Placed": C2},
                     title="Average Employability Score: Placed vs Not Placed")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(360, yaxis_range=[0, comp["Employability_Score"].max() * 1.2],
                                        xaxis_title="", yaxis_title="Avg Employability Score", showlegend=False))
        show(fig)
    col_e, col_f = st.columns(2)
    with col_e:
        fig = px.box(df, x="Placement_Status", y="CGPA", color="Placement_Status",
                     color_discrete_map={"Placed": C1, "Not Placed": C2}, title="CGPA Distribution: Placed vs Not Placed")
        fig.update_layout(**dark_layout(360, showlegend=False, xaxis_title="", yaxis_title="CGPA"))
        show(fig)
    with col_f:
        fig = px.box(df, x="Placement_Status", y="Employability_Score", color="Placement_Status",
                     color_discrete_map={"Placed": C1, "Not Placed": C2}, title="Employability Score Distribution: Placed vs Not Placed")
        fig.update_layout(**dark_layout(360, showlegend=False, xaxis_title="", yaxis_title="Employability Score"))
        show(fig)

    # ── Section 3: Salary Analysis
    st.markdown('<div class="section-header">💰 Salary Analysis</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_h = st.columns(2)
    with col_g:
        data = (placed_df.groupby("Company_Tier")["Starting_Salary_USD"]
                .mean().round(0).reset_index(name="Avg Starting Salary (USD)")
                .sort_values("Avg Starting Salary (USD)", ascending=False))
        fig = px.bar(data, x="Company_Tier", y="Avg Starting Salary (USD)", text="Avg Starting Salary (USD)",
                     color_discrete_sequence=[C1], title="Average Starting Salary by Company Tier")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(380, xaxis_title="Company Tier", yaxis_title="Avg Salary (USD)"))
        show(fig)
    with col_h:
        data = (placed_df.groupby("Major")["Starting_Salary_USD"]
                .mean().round(0).reset_index(name="Avg Starting Salary (USD)")
                .sort_values("Avg Starting Salary (USD)"))
        fig = px.bar(data, x="Avg Starting Salary (USD)", y="Major", orientation="h",
                     text="Avg Starting Salary (USD)", color_discrete_sequence=[CV],
                     title="Average Starting Salary by Major (Placed Students)")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(380, xaxis_title="Avg Salary (USD)", yaxis_title="",
                                        margin=dict(l=10, r=90, t=45, b=20)))
        show(fig)

    # ── Section 4: Experience & Skills Impact
    st.markdown('<div class="section-header">🛠️ Experience & Skills Impact on Placement</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_i, col_j = st.columns(2)
    with col_i:
        data = (df.groupby("Internships")["Placement_Status"]
                .apply(lambda s: round((s == "Placed").mean() * 100, 2))
                .reset_index(name="Placement Rate (%)").sort_values("Internships"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data["Internships"].astype(str), y=data["Placement Rate (%)"],
                             name="Placement Rate", marker_color=C1, opacity=0.8,
                             text=data["Placement Rate (%)"], texttemplate="%{text:.1f}%",
                             textposition="outside", textfont=dict(color=TEXT)))
        fig.add_trace(go.Scatter(x=data["Internships"].astype(str), y=data["Placement Rate (%)"],
                                 name="Trend", mode="lines+markers",
                                 line=dict(color=C2, width=2), marker=dict(size=7, color=C2)))
        fig.update_layout(**dark_layout(380, title="Placement Rate by Number of Internships",
                                        xaxis_title="Number of Internships", yaxis_title="Placement Rate (%)",
                                        yaxis_range=[0, 115],
                                        legend=dict(orientation="h", y=-0.18, bgcolor=BG,
                                                    bordercolor=GRID, font=dict(color=TEXT))))
        show(fig)
    with col_j:
        data = (df.groupby("GitHub_Profile")["Placement_Status"]
                .apply(lambda s: round((s == "Placed").mean() * 100, 2))
                .reset_index(name="Placement Rate (%)"))
        fig = px.bar(data, x="GitHub_Profile", y="Placement Rate (%)", text="Placement Rate (%)",
                     color="GitHub_Profile", color_discrete_map={"Yes": C1, "No": C2},
                     title="Placement Rate by GitHub Profile (Yes vs No)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(color=TEXT))
        fig.update_layout(**dark_layout(380, yaxis_range=[0, 100], xaxis_title="Has GitHub Profile",
                                        yaxis_title="Placement Rate (%)", showlegend=False))
        show(fig)
    data = (df.groupby("Certifications")["Placement_Status"]
            .apply(lambda s: round((s == "Placed").mean() * 100, 2))
            .reset_index(name="Placement Rate (%)").sort_values("Certifications"))
    fig = px.bar(data, x="Certifications", y="Placement Rate (%)", text="Placement Rate (%)",
                 color_discrete_sequence=[C1], title="Placement Rate by Number of Certifications")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(color=TEXT))
    fig.update_layout(**dark_layout(360, yaxis_range=[0, 110],
                                    xaxis_title="Number of Certifications", yaxis_title="Placement Rate (%)"))
    show(fig, w=1400, h=360)

    # ── Section 5: Distributions
    st.markdown('<div class="section-header">🍰 Distributions</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_k, col_l = st.columns(2)
    with col_k:
        counts = df["Placement_Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count", color="Status",
                     color_discrete_map={"Placed": C1, "Not Placed": C2},
                     title="Placement Status Distribution", hole=0.35)
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont=dict(color=TEXT))
        fig.update_layout(height=360, autosize=False, paper_bgcolor=BG, font=dict(color=TEXT),
                          title_font=dict(color="#f1f5f9"), margin=dict(l=20, r=20, t=45, b=20))
        show(fig)
    with col_l:
        counts = placed_df["Placement_Mode"].value_counts().reset_index()
        counts.columns = ["Placement Mode", "Count"]
        MODE_COLOURS = {"Campus Placement": "#14b8a6", "Off-Campus": "#f97316",
                        "Internship Conversion": "#a855f7", "Referral": "#22c55e",
                        "Not Applicable": "#64748b"}
        fig = px.pie(counts, names="Placement Mode", values="Count", color="Placement Mode",
                     color_discrete_map=MODE_COLOURS,
                     title="Placement Mode Distribution (Placed Students)", hole=0.35)
        fig.update_traces(textposition="outside", textinfo="percent+label", textfont=dict(color=TEXT))
        fig.update_layout(height=360, autosize=False, paper_bgcolor=BG, font=dict(color=TEXT),
                          title_font=dict(color="#f1f5f9"), margin=dict(l=20, r=20, t=45, b=20))
        show(fig)

    # ── Section 6: Correlation Heatmap
    st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    numeric_cols = ["CGPA", "Study_Hours_Per_Week", "Internships", "Certifications",
                    "Programming_Skill", "Projects_Completed", "Resume_Score",
                    "Communication_Skills", "Interview_Score", "Employability_Score",
                    "Starting_Salary_USD"]
    corr = df[numeric_cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="Blues", zmid=0, text=corr.values, texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT), hovertemplate="%{y} x %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlation Matrix of Key Numeric Features",
                      height=520, autosize=False, plot_bgcolor=BG, paper_bgcolor=BG,
                      font=dict(color=TEXT), title_font=dict(color="#f1f5f9", size=14),
                      xaxis=dict(tickangle=-45, tickfont=dict(color=MUTE), gridcolor=GRID),
                      yaxis=dict(tickfont=dict(color=MUTE), gridcolor=GRID),
                      margin=dict(l=140, r=20, t=60, b=20))
    show(fig, w=1400, h=520)

    # ── Section 7: Data Table
    st.markdown('<div class="section-header">🗂️ Data Table View</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    display_cols = ["Student_ID", "Age", "Gender", "University_Year", "Major",
                    "CGPA", "Internships", "Certifications", "GitHub_Profile",
                    "Employability_Score", "Interview_Score", "Placement_Status",
                    "Company_Tier", "Career_Field", "Placement_Mode", "Starting_Salary_USD"]
    st.markdown(f"Showing **{len(df):,}** records matching the current filter "
                f"({placement_rate}% placement rate).")
    st.dataframe(df[display_cols].reset_index(drop=True), height=420, use_container_width=True)
    st.download_button(label="⬇️ Download filtered data as CSV",
                       data=df[display_cols].to_csv(index=False).encode("utf-8"),
                       file_name="filtered_student_data.csv", mime="text/csv")
    st.markdown("---")
    st.markdown("<small style='color:#57606a'>PlacementIQ — Student Career Success Analysis | "
                "50,000 student records | Data Analytics Project</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — auto-detect mode
# ─────────────────────────────────────────────────────────────────────────────
#
# Streamlit sets up its script runner before executing this file.
# Checking for "streamlit.runtime.scriptrunner" in sys.modules is the
# reliable way to know whether we are inside a Streamlit process.
#
#   python main.py                        → run_analysis()  (terminal output)
#   python -m streamlit run main.py       → run_dashboard() (browser UI)

if "streamlit.runtime.scriptrunner" in sys.modules:
    # Running inside Streamlit — render the dashboard
    run_dashboard()
elif __name__ == "__main__":
    # Running as a plain Python script — run the analysis pipeline
    run_analysis()
