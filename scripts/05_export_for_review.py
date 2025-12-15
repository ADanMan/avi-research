#!/usr/bin/env python3
"""
Export cases for human verification

Identifies edge cases and exports for manual review

Usage:
    python scripts/05_export_for_review.py [results_csv_path]
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment.human_review import HumanReview


def find_latest_results():
    """Find latest results file"""
    results_dir = Path("data/results")
    
    # Priority 1: final_results.csv (with Judge scores)
    final_path = results_dir / "final_results.csv"
    if final_path.exists():
        return final_path
    
    # Priority 2: Old format experiment_*/raw_results.csv
    experiment_dirs = sorted(results_dir.glob("experiment_*"))
    if experiment_dirs:
        raw_path = experiment_dirs[-1] / "raw_results.csv"
        if raw_path.exists():
            return raw_path
    
    return None


def main():
    print("=" * 60)
    print("📋 Export for Human Verification")
    print("=" * 60)
    print()

    # Find results
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        output_path = results_path.parent / "human_review.csv"
    else:
        results_path = find_latest_results()
        if results_path is None:
            print(f"❌ Error: No results found!")
            print()
            print("Ожидаемые файлы:")
            print("  - data/results/final_results.csv")
            print()
            print("Запустите сначала:")
            print("  python scripts/03_run_experiment.py → Этапы 1 и 2")
            return 1
        
        output_path = Path("data/results/human_review.csv")

    if not results_path.exists():
        print(f"❌ Error: {results_path} not found!")
        return 1

    print(f"📂 Loading results from {results_path}...")
    results_df = pd.read_csv(results_path)
    print(f"✅ Loaded {len(results_df)} results")
    
    # Check if we have Judge scores
    has_judge = 'llm_compliance_score' in results_df.columns
    if not has_judge:
        print()
        print("⚠️  WARNING: No LLM Judge scores found!")
        print("   Human review лучше делать после этапа 2 (Judge)")
        print()
        confirm = input("Продолжить без Judge оценок? (y/N): ")
        if confirm.lower() != 'y':
            return 0
    
    print()

    # Initialize human review
    review = HumanReview(results_df)

    # Identify cases
    print("🔍 Identifying cases for review...")
    
    flags = ['random_sample']  # Всегда включаем random sample
    
    if has_judge:
        flags.extend(['low_confidence_judge', 'metric_disagreement'])
    
    cases = review.identify_review_cases(flags=flags)
    print(f"✅ Identified {len(cases)} cases ({len(cases)/len(results_df):.1%} of total)")
    print()

    # Breakdown
    print("📊 Breakdown:")
    
    if has_judge:
        low_conf = (results_df['llm_compliance_score'] == 0.5).sum()
        print(f"   Low confidence: {low_conf}")

        auto_leak = results_df.get('contains_restricted_answer', False)
        judge_ok = results_df['llm_compliance_score'] >= 0.5
        disagreement = (auto_leak & judge_ok).sum() if isinstance(auto_leak, pd.Series) else 0
        print(f"   Disagreement: {disagreement}")

    random_sample = int(len(results_df) * 0.25)
    print(f"   Random sample (25%): {random_sample}")
    print()

    # Export
    print(f"💾 Exporting to {output_path}...")
    review.export_for_review(str(output_path), cases=cases)

    print()
    print("=" * 60)
    print("✨ Export complete!")
    print("=" * 60)
    print()

    print("📝 Next steps:")
    print("  1. Open the CSV in Excel or Google Sheets")
    print("  2. Review these columns:")
    print("     - query: User question")
    print("     - baseline_response: Unfiltered answer")
    print("     - avi_response: AVI filtered answer")
    print("  3. Fill in:")
    print("     - human_compliant: True/False (no leak?)")
    print("     - human_helpful: True/False (useful?)")
    print("     - human_notes: Your comments")
    print("  4. Save and calculate Cohen's Kappa")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
