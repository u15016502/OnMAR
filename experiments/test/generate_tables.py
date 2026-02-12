"""
Table Generation Utility
========================

Generate publication-ready tables matching the thesis format.
Outputs LaTeX, CSV, and Markdown formats.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from scipy import stats


class TableGenerator:
    """Generate publication-ready tables."""

    def __init__(self, results_file: str):
        """
        Initialize table generator.

        Args:
            results_file: Path to results JSON file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)

    def generate_thesis_table_8_2(self, output_formats: List[str] = ['txt', 'csv', 'latex']):
        """
        Generate Table 8.2 from thesis: Average ranking per application.

        Args:
            output_formats: List of formats to generate ('txt', 'csv', 'latex', 'md')
        """
        # Process each dataset
        tables = {}

        for dataset, approaches in sorted(self.results.items()):
            rows = []

            # Calculate rankings using two-tailed Mann-Whitney U test
            approach_keys = list(approaches.keys())
            rankings = {}

            for key in approach_keys:
                if not approaches[key]:
                    continue

                # Get test performance values
                values = [r.get('test_results', {}).get('test_performance', 0)
                         for r in approaches[key]]

                # Compare against all other approaches
                rank_sum = 0
                comparisons = 0

                for other_key in approach_keys:
                    if other_key == key or not approaches[other_key]:
                        continue

                    other_values = [r.get('test_results', {}).get('test_performance', 0)
                                   for r in approaches[other_key]]

                    # Two-tailed test for equivalence
                    _, p_value = stats.mannwhitneyu(values, other_values, alternative='two-sided')

                    if p_value < 0.05:
                        # Statistically different - assign rank based on mean
                        if np.mean(values) > np.mean(other_values):
                            rank_sum += 1  # Better than other
                        else:
                            rank_sum += 2  # Worse than other
                    else:
                        # Statistically equivalent - both get same rank
                        rank_sum += 1.5

                    comparisons += 1

                if comparisons > 0:
                    rankings[key] = rank_sum / comparisons
                else:
                    rankings[key] = 1.0

            # Build rows for this dataset
            for approach_key in sorted(approach_keys, key=lambda x: rankings.get(x, 999)):
                runs = approaches[approach_key]
                if not runs:
                    continue

                values = [r.get('test_results', {}).get('test_performance', 0) for r in runs]

                rows.append({
                    'Dataset': dataset.upper(),
                    'Approach': runs[0]['approach'],
                    'Meta-Learner': runs[0]['meta_learner'],
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Rank': rankings.get(approach_key, 999)
                })

            tables[dataset] = pd.DataFrame(rows)

        # Save in requested formats
        output_dir = Path('experiments/test/results/tables')
        output_dir.mkdir(parents=True, exist_ok=True)

        for dataset, df in tables.items():
            if 'txt' in output_formats:
                output_file = output_dir / f'table_8_2_{dataset}.txt'
                with open(output_file, 'w') as f:
                    f.write(f"Table 8.2 Format: {dataset.upper()}\n")
                    f.write("="*80 + "\n")
                    f.write(df.to_string(index=False))
                print(f"Text table saved: {output_file}")

            if 'csv' in output_formats:
                output_file = output_dir / f'table_8_2_{dataset}.csv'
                df.to_csv(output_file, index=False)
                print(f"CSV table saved: {output_file}")

            if 'latex' in output_formats:
                output_file = output_dir / f'table_8_2_{dataset}.tex'
                latex_str = df.to_latex(
                    index=False,
                    float_format="%.3f",
                    caption=f"Results for {dataset.upper()} dataset",
                    label=f"tab:{dataset}"
                )
                with open(output_file, 'w') as f:
                    f.write(latex_str)
                print(f"LaTeX table saved: {output_file}")

            if 'md' in output_formats:
                output_file = output_dir / f'table_8_2_{dataset}.md'
                with open(output_file, 'w') as f:
                    f.write(df.to_markdown(index=False))
                print(f"Markdown table saved: {output_file}")

    def generate_performance_summary(self, metric: str = 'test_performance'):
        """
        Generate summary table with all key metrics.

        Args:
            metric: Primary metric to summarize
        """
        rows = []

        for dataset, approaches in sorted(self.results.items()):
            for approach_key, runs in sorted(approaches.items()):
                if not runs:
                    continue

                # Extract all metrics
                if metric == 'test_performance':
                    perf_values = [r.get('test_results', {}).get('test_performance', 0) for r in runs]
                else:
                    perf_values = [r.get(metric, 0) for r in runs]

                runtime_values = [r.get('total_time', 0) / 60.0 for r in runs]  # minutes

                # Reuse metrics (only for MAR approaches)
                if 'design_reuse_percentage' in runs[0]:
                    reuse_pct = [r['design_reuse_percentage'] for r in runs]
                    design_calls = [r['num_design_algorithm_calls'] for r in runs]
                else:
                    reuse_pct = [0]
                    design_calls = [0]

                rows.append({
                    'Dataset': dataset.upper(),
                    'Approach': runs[0]['approach'],
                    'Meta-Learner': runs[0]['meta_learner'],
                    f'{metric} (mean)': np.mean(perf_values),
                    f'{metric} (std)': np.std(perf_values),
                    'Runtime (min mean)': np.mean(runtime_values),
                    'Runtime (min std)': np.std(runtime_values),
                    'Reuse % (mean)': np.mean(reuse_pct),
                    'Design Calls (mean)': np.mean(design_calls),
                    'N runs': len(runs)
                })

        df = pd.DataFrame(rows)

        # Save
        output_dir = Path('experiments/test/results/tables')
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV format
        output_file = output_dir / 'performance_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\nPerformance summary saved: {output_file}")

        # LaTeX format
        output_file = output_dir / 'performance_summary.tex'
        latex_str = df.to_latex(index=False, float_format="%.4f")
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX summary saved: {output_file}")

        # Markdown format
        output_file = output_dir / 'performance_summary.md'
        with open(output_file, 'w') as f:
            f.write("# Performance Summary\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Markdown summary saved: {output_file}")

        return df

    def generate_statistical_comparison_table(self):
        """
        Generate pairwise statistical comparison table.
        Shows Mann-Whitney U test results for all approach pairs.
        """
        rows = []

        for dataset, approaches in sorted(self.results.items()):
            approach_keys = list(approaches.keys())

            for i, key1 in enumerate(approach_keys):
                for key2 in approach_keys[i+1:]:
                    runs1 = approaches.get(key1, [])
                    runs2 = approaches.get(key2, [])

                    if not runs1 or not runs2:
                        continue

                    values1 = [r.get('test_results', {}).get('test_performance', 0) for r in runs1]
                    values2 = [r.get('test_results', {}).get('test_performance', 0) for r in runs2]

                    # Two-sided test
                    stat, p_value_two = stats.mannwhitneyu(values1, values2, alternative='two-sided')

                    # One-sided test (key1 > key2)
                    _, p_value_greater = stats.mannwhitneyu(values1, values2, alternative='greater')

                    # Determine relationship
                    if p_value_two >= 0.05:
                        relationship = "≈"
                    elif p_value_greater < 0.05:
                        relationship = ">"
                    else:
                        relationship = "<"

                    rows.append({
                        'Dataset': dataset.upper(),
                        'Approach 1': key1,
                        'Approach 2': key2,
                        'Mean 1': np.mean(values1),
                        'Mean 2': np.mean(values2),
                        'Relationship': relationship,
                        'p-value (two-tailed)': p_value_two,
                        'p-value (one-tailed)': p_value_greater
                    })

        df = pd.DataFrame(rows)

        # Save
        output_dir = Path('experiments/test/results/tables')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / 'statistical_comparisons.csv'
        df.to_csv(output_file, index=False)
        print(f"\nStatistical comparisons saved: {output_file}")

        # Also save significant results only
        sig_df = df[df['p-value (two-tailed)'] < 0.05]
        output_file = output_dir / 'statistical_comparisons_significant.csv'
        sig_df.to_csv(output_file, index=False)
        print(f"Significant comparisons saved: {output_file}")

        return df


def main():
    """Generate all tables."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication tables")
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['txt', 'csv', 'latex', 'md'],
        help='Output formats (txt, csv, latex, md)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = TableGenerator(args.results_file)

    print("="*80)
    print("GENERATING PUBLICATION TABLES")
    print("="*80)

    # Generate Table 8.2 format
    print("\nGenerating Thesis Table 8.2 format...")
    generator.generate_thesis_table_8_2(output_formats=args.formats)

    # Generate performance summary
    print("\nGenerating performance summary table...")
    generator.generate_performance_summary()

    # Generate statistical comparisons
    print("\nGenerating statistical comparison table...")
    generator.generate_statistical_comparison_table()

    print("\n" + "="*80)
    print("TABLE GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
