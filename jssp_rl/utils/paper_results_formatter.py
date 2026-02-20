"""paper_results_formatter.py - Generate publication-quality results tables
==========================================================================
Produces Markdown + LaTeX tables split by:
  - In-Distribution (trained sizes)
  - Generalization (unseen sizes)

Grouped by benchmark (FT, Taillard, DMU) with BKS comparison.
"""

import pandas as pd
import os
import json


def generate_paper_table(training_logs_dir, hybrid_results_path, output_path,
                         benchmark_results_dir=None):
    """
    Aggregates training metrics and benchmark/hybrid results into
    publication-quality tables split by In-Distribution vs Generalization.
    """
    print(f"[Paper Formatter] Generating results to {output_path}...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if benchmark_results_dir is None:
        benchmark_results_dir = os.path.join(base_dir, "eval")

    with open(output_path, "w") as f:
        f.write("# JSSP Research Results Summary\n\n")
        
        # ============================================================
        # Section 1: Training Phase Summary
        # ============================================================
        f.write("## 1. Training Phase Summary\n\n")
        all_train_data = []
        if os.path.exists(training_logs_dir):
            files = [fn for fn in os.listdir(training_logs_dir)
                     if fn.startswith("training_log_") and fn.endswith(".csv")]
            for fn in sorted(files):
                try:
                    df = pd.read_csv(os.path.join(training_logs_dir, fn))
                    best_idx = df['val_makespan_best_of_k'].idxmin()
                    best_row = df.loc[best_idx].copy()
                    all_train_data.append(best_row)
                except Exception:
                    pass

        if all_train_data:
            train_summary = pd.DataFrame(all_train_data)
            cols = [c for c in ['phase', 'epoch', 'val_makespan_greedy',
                                'val_makespan_best_of_k', 'epoch_time_seconds']
                    if c in train_summary.columns]
            f.write(train_summary[cols].to_markdown(index=False))
        else:
            f.write("No training data found.\n")

        # ============================================================
        # Section 2: RL-Only Benchmark (Split by Distribution)
        # ============================================================
        combined_path = os.path.join(benchmark_results_dir, "benchmark_all_results.csv")
        if os.path.exists(combined_path):
            bm_df = pd.read_csv(combined_path)
            
            bok_col = [c for c in bm_df.columns if c.startswith("rl_best_of_")]
            gap_col = [c for c in bm_df.columns if c.startswith("gap_best_of_")]
            bok_col = bok_col[0] if bok_col else "rl_greedy"
            gap_col = gap_col[0] if gap_col else "gap_greedy_pct"
            
            for dist_label in ["In-Distribution", "Generalization"]:
                dist_df = bm_df[bm_df["distribution"] == dist_label] if "distribution" in bm_df.columns else bm_df
                if dist_df.empty:
                    continue
                
                f.write(f"\n\n## 2{'a' if dist_label == 'In-Distribution' else 'b'}. RL Evaluation — {dist_label}\n\n")
                
                for bm_name in dist_df["benchmark"].unique():
                    subset = dist_df[dist_df["benchmark"] == bm_name]
                    f.write(f"\n### {bm_name}\n\n")
                    
                    display_cols = ["instance", "size", "bks", "rl_greedy", bok_col, gap_col]
                    display_cols = [c for c in display_cols if c in subset.columns]
                    f.write(subset[display_cols].to_markdown(index=False))
                    
                    f.write("\n\n**Averages:**\n\n")
                    size_summary = subset.groupby("size").agg({
                        "bks": "mean", "rl_greedy": "mean",
                        bok_col: "mean", gap_col: "mean",
                    }).round(1)
                    size_summary.columns = ["BKS", "Greedy", "Best-of-K", "Gap(%)"]
                    f.write(size_summary.to_markdown())
                    f.write("\n")

        # ============================================================
        # Section 3: Hybrid (Split by Distribution)
        # ============================================================
        if os.path.exists(hybrid_results_path):
            hybrid_df = pd.read_csv(hybrid_results_path)
            
            for dist_label in ["In-Distribution", "Generalization"]:
                dist_df = hybrid_df[hybrid_df["distribution"] == dist_label] if "distribution" in hybrid_df.columns else hybrid_df
                if dist_df.empty:
                    continue
                
                f.write(f"\n\n## 3{'a' if dist_label == 'In-Distribution' else 'b'}. Hybrid RL+CP — {dist_label}\n\n")
                
                for bm_name in dist_df["benchmark"].unique():
                    subset = dist_df[dist_df["benchmark"] == bm_name]
                    f.write(f"\n### {bm_name}\n\n")
                    
                    cols = ["instance", "size", "bks", "rl_ms", "cp_cold_ms", "hybrid_ms",
                            "gap_rl_pct", "gap_cp_pct", "gap_hybrid_pct", "best_method"]
                    cols = [c for c in cols if c in subset.columns]
                    f.write(subset[cols].to_markdown(index=False))
                    
                    f.write("\n\n**Averages:**\n\n")
                    agg_cols = {
                        "bks": "mean", "rl_ms": "mean", "cp_cold_ms": "mean", "hybrid_ms": "mean",
                        "gap_rl_pct": "mean", "gap_cp_pct": "mean", "gap_hybrid_pct": "mean",
                    }
                    agg_cols = {k: v for k, v in agg_cols.items() if k in subset.columns}
                    summary = subset.groupby("size").agg(agg_cols).round(1)
                    f.write(summary.to_markdown())
                    f.write("\n")
            
            # Winner counts
            if "best_method" in hybrid_df.columns:
                f.write("\n### Method Win Counts\n\n")
                if "distribution" in hybrid_df.columns:
                    wins = hybrid_df.groupby("distribution")["best_method"].value_counts().unstack(fill_value=0)
                    f.write(wins.to_markdown())
                else:
                    wins = hybrid_df["best_method"].value_counts()
                    f.write(wins.to_markdown())
                f.write("\n")

        # ============================================================
        # Section 4: LaTeX Tables
        # ============================================================
        f.write("\n\n## 4. LaTeX Tables\n\n")
        
        if os.path.exists(hybrid_results_path):
            hybrid_df = pd.read_csv(hybrid_results_path)
            
            for dist_label in ["In-Distribution", "Generalization"]:
                dist_df = hybrid_df[hybrid_df["distribution"] == dist_label] if "distribution" in hybrid_df.columns else hybrid_df
                if dist_df.empty:
                    continue
                
                f.write(f"\n### {dist_label} LaTeX\n\n")
                agg = dist_df.groupby("size").agg({
                    "bks": "mean", "rl_ms": "mean", "cp_cold_ms": "mean", "hybrid_ms": "mean",
                    "gap_rl_pct": "mean", "gap_cp_pct": "mean", "gap_hybrid_pct": "mean",
                }).round(1)
                
                caption = "In-Distribution" if dist_label == "In-Distribution" else "Generalization (Unseen Sizes)"
                
                f.write("```latex\n")
                f.write("\\begin{table}[h]\n\\centering\n")
                f.write(f"\\caption{{Results — {caption}}}\n")
                f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
                f.write("Size & BKS & RL & CP & Hybrid & Gap$_{RL}$ & Gap$_{CP}$ & Gap$_{Hyb}$ \\\\\n")
                f.write("\\midrule\n")
                for size, row in agg.iterrows():
                    f.write(f"{size} & {row['bks']:.0f} & {row['rl_ms']:.0f} & "
                            f"{row['cp_cold_ms']:.0f} & {row['hybrid_ms']:.0f} & "
                            f"{row['gap_rl_pct']:.1f}\\% & {row['gap_cp_pct']:.1f}\\% & "
                            f"{row['gap_hybrid_pct']:.1f}\\% \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
                f.write("```\n")

    print(f"[Success] Report saved to {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    generate_paper_table(
        training_logs_dir=os.path.join(base, "outputs", "logs"),
        hybrid_results_path=os.path.join(base, "eval", "hybrid_experiment_results.csv"),
        output_path=os.path.join(base, "outputs", "final_paper_results.md"),
        benchmark_results_dir=os.path.join(base, "eval"),
    )
