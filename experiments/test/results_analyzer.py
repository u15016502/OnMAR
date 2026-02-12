"""
Results Analysis Utility
=========================

Utilities for parsing, analyzing, and visualizing experimental results
in the format used in the thesis.

This script can:
1. Generate statistical comparisons (Mann-Whitney U tests)
2. Create box plots matching Figure 8.7 format
3. Generate ranking tables matching Table 8.2 format
4. Calculate effect sizes and confidence intervals
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing experimental results
        """
        self.results_dir = Path(results_dir)
        self.results = None

    def load_results(self, results_file: str = None):
        """
        Load results from JSON file.

        Args:
            results_file: Specific results file to load. If None, loads most recent.
        """
        if results_file is None:
            # Find most recent results file
            all_results = list(self.results_dir.glob("all_results_*.json"))
            if not all_results:
                raise FileNotFoundError(f"No results files found in {self.results_dir}")
            results_file = max(all_results, key=lambda p: p.stat().st_mtime)
        else:
            results_file = self.results_dir / results_file

        print(f"Loading results from: {results_file}")

        with open(results_file, 'r') as f:
            self.results = json.load(f)

        print(f"Loaded results for {len(self.results)} datasets")

    def mann_whitney_comparison(
        self,
        dataset: str,
        approach1: str,
        approach2: str,
        metric: str = 'test_performance'
    ) -> Tuple[float, float, str]:
        """
        Perform Mann-Whitney U test between two approaches.

        Args:
            dataset: Dataset name
            approach1: First approach key
            approach2: Second approach key
            metric: Metric to compare

        Returns:
            Tuple of (U-statistic, p-value, interpretation)
        """
        if dataset not in self.results:
            raise ValueError(f"Dataset {dataset} not found in results")

        runs1 = self.results[dataset].get(approach1, [])
        runs2 = self.results[dataset].get(approach2, [])

        if not runs1 or not runs2:
            return None, None, "Missing data"

        # Extract metric values
        if metric == 'test_performance':
            values1 = [r.get('test_results', {}).get('test_performance', 0) for r in runs1]
            values2 = [r.get('test_results', {}).get('test_performance', 0) for r in runs2]
        else:
            values1 = [r.get(metric, 0) for r in runs1]
            values2 = [r.get(metric, 0) for r in runs2]

        # Perform one-tailed test (approach1 > approach2)
        statistic, p_value_two = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        _, p_value_greater = stats.mannwhitneyu(values1, values2, alternative='greater')

        # Interpret results
        if p_value_greater < 0.05:
            interpretation = f"{approach1} > {approach2} (p={p_value_greater:.4f})"
        elif p_value_two >= 0.05:
            interpretation = f"{approach1} ≈ {approach2} (p={p_value_two:.4f})"
        else:
            interpretation = f"{approach1} < {approach2} (p={1-p_value_greater:.4f})"

        return statistic, p_value_greater, interpretation

    def generate_ranking_table(
        self,
        dataset: str,
        metric: str = 'test_performance'
    ) -> pd.DataFrame:
        """
        Generate ranking table in thesis Table 8.2 format.

        Args:
            dataset: Dataset name
            metric: Metric to rank by

        Returns:
            DataFrame with rankings
        """
        if dataset not in self.results:
            raise ValueError(f"Dataset {dataset} not found")

        rankings = []

        for approach_key, runs in self.results[dataset].items():
            if not runs:
                continue

            # Extract metric
            if metric == 'test_performance':
                values = [r.get('test_results', {}).get('test_performance', 0) for r in runs]
            else:
                values = [r.get(metric, 0) for r in runs]

            rankings.append({
                'Approach': runs[0]['approach'],
                'Meta-Learner': runs[0]['meta_learner'],
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Median': np.median(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': len(values)
            })

        df = pd.DataFrame(rankings)
        df = df.sort_values('Mean', ascending=False)
        df['Rank'] = range(1, len(df) + 1)

        return df

    def generate_boxplots(
        self,
        dataset: str,
        metric: str = 'test_performance',
        save_path: str = None
    ):
        """
        Generate box plots matching thesis Figure 8.7 format.

        Args:
            dataset: Dataset name
            metric: Metric to plot
            save_path: Path to save figure (if None, displays plot)
        """
        if dataset not in self.results:
            raise ValueError(f"Dataset {dataset} not found")

        # Prepare data
        data = []
        labels = []

        # Order: OnMAR-Acc, OnMAR-Des, OffMAR-Acc, OffMAR-Des, AutoSklearn
        # For each meta-learner: kNN, RF, XGBoost
        meta_learners = ['knn', 'rf', 'xgboost']

        for approach in ['OnMAR-Acc', 'OnMAR-Des', 'OffMAR-Acc', 'OffMAR-Des']:
            for ml in meta_learners:
                key = f"{approach}-{ml}"
                runs = self.results[dataset].get(key, [])

                if runs:
                    if metric == 'test_performance':
                        values = [r.get('test_results', {}).get('test_performance', 0) for r in runs]
                    else:
                        values = [r.get(metric, 0) for r in runs]

                    data.append(values)
                    labels.append(f"{approach}\n{ml.upper()}")

        # Add AutoSklearn
        autosklearn_runs = self.results[dataset].get('AutoSklearn', [])
        if autosklearn_runs:
            if metric == 'test_performance':
                values = [r.get('test_results', {}).get('test_performance', 0) for r in autosklearn_runs]
            else:
                values = [r.get(metric, 0) for r in autosklearn_runs]
            data.append(values)
            labels.append("AutoSklearn")

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 6))

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes by approach
        colors = []
        for label in labels:
            if 'OnMAR-Acc' in label:
                colors.append('lightblue')
            elif 'OnMAR-Des' in label:
                colors.append('lightcoral')
            elif 'OffMAR-Acc' in label:
                colors.append('lightgreen')
            elif 'OffMAR-Des' in label:
                colors.append('lightyellow')
            else:  # AutoSklearn
                colors.append('orange')

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Testing Accuracy', fontsize=12)
        ax.set_title(f'CNN Configuration Results - {dataset.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_comprehensive_report(self, output_file: str = None):
        """
        Generate comprehensive analysis report.

        Args:
            output_file: Output file path (if None, prints to console)
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        output = []
        output.append("="*80)
        output.append("COMPREHENSIVE EXPERIMENTAL RESULTS ANALYSIS")
        output.append("="*80)
        output.append("")

        for dataset in sorted(self.results.keys()):
            output.append(f"\nDATASET: {dataset.upper()}")
            output.append("-"*80)

            # Generate ranking table
            ranking_df = self.generate_ranking_table(dataset)

            output.append("\nRanking by Test Performance:")
            output.append(ranking_df.to_string(index=False))

            # Statistical comparisons
            output.append("\n\nStatistical Comparisons (Mann-Whitney U):")
            output.append("-"*80)

            # Compare OnMAR-Acc vs OffMAR-Acc for each meta-learner
            for ml in ['knn', 'rf', 'xgboost']:
                _, p, interp = self.mann_whitney_comparison(
                    dataset,
                    f'OnMAR-Acc-{ml}',
                    f'OffMAR-Acc-{ml}'
                )
                if p is not None:
                    output.append(f"  {interp}")

            # Compare best OnMAR vs AutoSklearn
            best_onmar = ranking_df[ranking_df['Approach'].str.contains('OnMAR')].iloc[0]
            output.append(f"\nBest approach: {best_onmar['Approach']} with {best_onmar['Meta-Learner']}")
            output.append(f"Mean performance: {best_onmar['Mean']:.4f} ± {best_onmar['Std']:.4f}")

            output.append("\n")

        report_text = "\n".join(output)

        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        else:
            print(report_text)

    def calculate_effect_sizes(self, dataset: str) -> pd.DataFrame:
        """
        Calculate Cohen's d effect sizes for all pairwise comparisons.

        Args:
            dataset: Dataset name

        Returns:
            DataFrame with effect sizes
        """
        if dataset not in self.results:
            raise ValueError(f"Dataset {dataset} not found")

        approaches = list(self.results[dataset].keys())
        effect_sizes = []

        for i, app1 in enumerate(approaches):
            for app2 in approaches[i+1:]:
                runs1 = self.results[dataset][app1]
                runs2 = self.results[dataset][app2]

                if not runs1 or not runs2:
                    continue

                values1 = [r.get('test_results', {}).get('test_performance', 0) for r in runs1]
                values2 = [r.get('test_results', {}).get('test_performance', 0) for r in runs2]

                # Calculate Cohen's d
                mean_diff = np.mean(values1) - np.mean(values2)
                pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

                effect_sizes.append({
                    'Approach 1': app1,
                    'Approach 2': app2,
                    'Mean Diff': mean_diff,
                    "Cohen's d": cohens_d,
                    'Effect Size': self._interpret_cohens_d(cohens_d)
                })

        return pd.DataFrame(effect_sizes)

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='experiments/test/results',
        help='Results directory'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate box plots for all datasets'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive report'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)

    # Load results
    try:
        analyzer.load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Generate report
    if args.generate_report:
        analyzer.generate_comprehensive_report(output_file="analysis_report.txt")

    # Generate plots
    if args.generate_plots:
        plots_dir = Path(args.results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        for dataset in analyzer.results.keys():
            try:
                save_path = plots_dir / f"{dataset}_boxplot.png"
                analyzer.generate_boxplots(dataset, save_path=str(save_path))
            except Exception as e:
                print(f"Error generating plot for {dataset}: {e}")


if __name__ == "__main__":
    main()
