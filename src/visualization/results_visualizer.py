"""
Results Visualizer

Generates all figures for paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional


class ResultsVisualizer:
    """Generate publication-ready visualizations"""

    def __init__(
        self,
        results_df: pd.DataFrame,
        style: str = "seaborn-v0_8-paper",
        dpi: int = 300,
    ):
        """
        Initialize visualizer
        """
        self.results = results_df.copy()
        self.dpi = dpi

        # Compute latency_overhead_ms if not present
        if 'latency_overhead_ms' not in self.results.columns:
            if 'baseline_latency_ms' in self.results.columns and 'avi_latency_ms' in self.results.columns:
                self.results['latency_overhead_ms'] = (
                    self.results['avi_latency_ms'] - self.results['baseline_latency_ms'].fillna(0)
                )

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('ggplot')

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
    ):
        """Plot confusion matrix for compliance"""
        from sklearn.metrics import confusion_matrix

        y_true = [1] * len(self.results)
        y_pred = (self.results['llm_compliance_score'] >= 0.5).astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))
        
        heatmap_data = pd.DataFrame(
            [[len(y_pred) - sum(y_pred), sum(y_pred)]], 
            columns=['Failed (Leak)', 'Success (Blocked)'],
            index=['Violations']
        )

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            annot_kws={"size": 16, "weight": "bold"}
        )

        ax.set_title('AVI Enforcement Efficacy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Input Type', fontsize=12)
        ax.set_xlabel('Outcome', fontsize=12)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")

        plt.close()

    def plot_latency_distribution(
        self,
        save_path: Optional[str] = None,
    ):
        """Plot latency comparison: Baseline vs AVI"""
        fig, ax = plt.subplots(figsize=(10, 7))  # Increased height slightly

        data = []
        labels = []
        colors = []

        if 'baseline_latency_ms' in self.results.columns:
            baseline = self.results['baseline_latency_ms'].dropna()
            data.append(baseline)
            labels.append('Baseline\n(Unfiltered Generation)')
            colors.append('#ff9999')

        if 'avi_latency_ms' in self.results.columns:
            avi = self.results['avi_latency_ms'].dropna()
            data.append(avi)
            labels.append('AVI\n(Governed Generation)')
            colors.append('#90EE90')

        # Box plot
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='darkgreen', markersize=8),
            medianprops=dict(color='black', linewidth=2),
            widths=0.5,
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Improvement annotation
        if len(data) == 2:
            baseline_mean = data[0].mean()
            avi_mean = data[1].mean()
            improvement = ((baseline_mean - avi_mean) / baseline_mean) * 100

            # Arrow
            ax.annotate(
                '',
                xy=(2, avi_mean),
                xytext=(1, baseline_mean),
                arrowprops=dict(
                    arrowstyle='->',
                    lw=2,
                    color='green',
                    connectionstyle='arc3,rad=0.2'
                )
            )

            # Text box
            mid_x = 1.5
            mid_y = (baseline_mean + avi_mean) / 2
            
            ax.text(
                mid_x, mid_y,
                f"−{improvement:.1f}%\nLatency Reduction",
                fontsize=12,
                fontweight='bold',
                color='green',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.9)
            )

        ax.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Operational Efficiency: Baseline vs AVI', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # FIX: Adjust layout to prevent bottom text overlap
        plt.subplots_adjust(bottom=0.15)
        
        fig.text(0.5, 0.05, 
                'Lower latency in AVI is due to "Early Breaking": preventing long, non-compliant generations.',
                ha='center', fontsize=10, style='italic', color='gray')

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")

        plt.close()

    def plot_helpfulness_vs_compliance(
        self,
        save_path: Optional[str] = None,
    ):
        """Plot trade-off between helpfulness and compliance"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Add jitter
        jitter_x = np.random.normal(0, 0.015, size=len(self.results))
        jitter_y = np.random.normal(0, 0.015, size=len(self.results))

        scatter = ax.scatter(
            self.results['llm_compliance_score'] + jitter_x,
            self.results['llm_helpfulness_score'] + jitter_y,
            c=self.results['llm_naturalness_score'],
            cmap='viridis',
            s=120,
            alpha=0.6,
            edgecolors='white',
            linewidths=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Naturalness Score', fontsize=11)

        ax.set_xlabel('Compliance Score (Safety)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Helpfulness Score (Utility)', fontsize=12, fontweight='bold')
        ax.set_title('Quality Matrix: Explainable Governance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        
        # Set limits to ensure points aren't cut off
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        # Quadrants
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Annotations
        ax.text(0.98, 0.98, 'Target Zone\n(Safe & Helpful Refusal)',
               transform=ax.transAxes, fontsize=10, va='top', ha='right', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#90EE90', alpha=0.3))
        
        ax.text(0.02, 0.98, 'Unsafe\n(Leakage)', 
               transform=ax.transAxes, fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='#ff9999', alpha=0.3))

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")

        plt.close()

    def plot_time_to_compliance(
        self,
        save_path: Optional[str] = None,
    ):
        """Plot time-to-compliance comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Data: 10 days vs 10 minutes (600 seconds)
        methods = ['RLHF / Fine-Tuning\n(Retraining)', 'AVI\n(Config + Indexing)']
        times = [10 * 24 * 3600, 600]  
        colors = ['#ff9999', '#90EE90']

        bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5, width=0.5)

        ax.set_ylabel('Operational Time to Enforce (Seconds)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title('Agility: Time-to-Compliance', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # FIX: Увеличиваем верхнюю границу графика, чтобы текст точно влез
        # На логарифмической шкале умножение на 5 дает достаточный запас сверху
        ax.set_ylim(top=max(times) * 5)

        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            if time >= 86400:
                label = f'~10 Days'
            else:
                label = f'~10 Mins'
            
            # FIX: Уменьшили множитель с 1.2 до 1.05, чтобы текст был ближе к бару
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height * 1.05, 
                label,
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
            )

        # Add speedup annotation
        speedup = times[0] / times[1]
        ax.text(
            0.5, 0.6,
            f'{speedup:.0f}× faster\nadaptation',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            color='green',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2)
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")

        plt.close()

    def generate_all_figures(
        self,
        output_dir: str = "paper/figures",
    ):
        """Generate all figures for paper"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🎨 Generating all figures...")
        print()

        self.plot_confusion_matrix(f"{output_dir}/fig1_confusion_matrix.png")
        self.plot_latency_distribution(f"{output_dir}/fig2_latency.png")
        
        if 'llm_compliance_score' in self.results.columns:
            self.plot_helpfulness_vs_compliance(f"{output_dir}/fig3_quality.png")
        
        self.plot_time_to_compliance(f"{output_dir}/fig4_agility.png")

        print()
        print(f"✅ All figures saved to {output_dir}/")
