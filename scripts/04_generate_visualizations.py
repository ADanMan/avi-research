#!/usr/bin/env python3
"""
Generate all visualizations for paper

Creates figures and tables from experiment results

Usage:
    python scripts/04_generate_visualizations.py [results_csv_path]
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.results_visualizer import ResultsVisualizer
from src.visualization.paper_tables import PaperTables


def find_latest_results():
    """Find latest results file in priority order"""
    results_dir = Path("data/results")
    
    # Priority 1: final_results.csv (with Judge scores)
    final_path = results_dir / "final_results.csv"
    if final_path.exists():
        return final_path, True
    
    # Priority 2: queries_results.csv (without Judge)
    queries_path = results_dir / "queries_results.csv"
    if queries_path.exists():
        return queries_path, False
    
    # Priority 3: Old format experiment_*/raw_results.csv
    experiment_dirs = sorted(results_dir.glob("experiment_*"))
    if experiment_dirs:
        raw_path = experiment_dirs[-1] / "raw_results.csv"
        if raw_path.exists():
            has_judge = 'llm_compliance_score' in pd.read_csv(raw_path, nrows=0).columns
            return raw_path, has_judge
    
    return None, False


def main():
    print("=" * 60)
    print("🎨 Generate Visualizations for Paper")
    print("=" * 60)
    print()

    # Find results
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
        if not results_path.exists():
            print(f"❌ Error: {results_path} not found!")
            return 1
        has_judge = 'llm_compliance_score' in pd.read_csv(results_path, nrows=0).columns
    else:
        results_path, has_judge = find_latest_results()
        if results_path is None:
            print(f"❌ Error: No results found!")
            print()
            print("Ожидаемые файлы:")
            print("  - data/results/final_results.csv (с Judge оценками)")
            print("  - data/results/queries_results.csv (без Judge)")
            print()
            print("Запустите сначала:")
            print("  python scripts/03_run_experiment.py")
            return 1

    print(f"📂 Loading results from {results_path}...")
    results_df = pd.read_csv(results_path)
    print(f"✅ Loaded {len(results_df)} results")
    
    if has_judge:
        print(f"✅ С LLM Judge оценками")
    else:
        print(f"⚠️  Без LLM Judge оценок (только automatic metrics)")
        print(f"   Для полной визуализации запустите этап 2 (Judge)")
    print()

    # Generate figures
    print("🎨 Generating figures...")
    print()

    viz = ResultsVisualizer(results_df, dpi=300)
    
    if has_judge:
        viz.generate_all_figures(output_dir="paper/figures")
    else:
        # Ограниченная визуализация без Judge
        print("⚠️  Генерация ограниченного набора графиков (нет Judge данных)")
        viz.plot_latency_distribution("paper/figures/fig2_latency.png")
        print("   ✓ Latency distribution")

    print()

    # Generate tables
    print("📊 Generating tables...")
    print()

    tables = PaperTables(results_df)
    
    if has_judge:
        tables.generate_all_tables(output_dir="paper/tables")
    else:
        print("⚠️  Таблицы требуют Judge оценок - пропущено")

    print()
    print("=" * 60)
    print("✨ Visualizations generated!")
    print("=" * 60)
    print()

    print("📁 Outputs:")
    print("   Figures: paper/figures/")
    if has_judge:
        print("   Tables:  paper/tables/")
    print()

    if not has_judge:
        print("💡 Для полной визуализации:")
        print("   1. Переключите на ВНЕШНИЙ VPN")
        print("   2. Запустите: python scripts/03_run_experiment.py → Этап 2")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
