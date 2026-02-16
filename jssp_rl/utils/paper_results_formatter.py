import pandas as pd
import os

def generate_paper_table(training_logs_dir, hybrid_results_path, output_path):
    """
    Aggregates training metrics and hybrid experiment results into a single
    professional table for publication.
    """
    print(f"[Paper Formatter] Aggregating results into {output_path}...")
    
    # 1. Aggregate Training Metrics
    all_train_data = []
    if os.path.exists(training_logs_dir):
        files = [f for f in os.listdir(training_logs_dir) if f.startswith("training_log_") and f.endswith(".csv")]
        for f in sorted(files):
            df = pd.read_csv(os.path.join(training_logs_dir, f))
            # Get best performance per phase
            best_idx = df['val_makespan_best_of_k'].idxmin()
            best_row = df.loc[best_idx].copy()
            all_train_data.append(best_row)
    
    train_summary = pd.DataFrame(all_train_data)
    
    # 2. Process Hybrid Results
    if os.path.exists(hybrid_results_path):
        hybrid_df = pd.read_csv(hybrid_results_path)
        
        # Calculate improvements
        hybrid_df['gain_vs_cold_ms'] = hybrid_df['cp_cold_ms'] - hybrid_df['cp_warm_ms']
        hybrid_df['speedup_vs_cold'] = hybrid_df['cp_cold_time'] / (hybrid_df['hybrid_time'] + 1e-6)
        
        # Create a summary row for Taillard averages
        avg_row = {
            "id": "AVERAGE",
            "size": "Mixed",
            "rl_ms": hybrid_df['rl_ms'].mean(),
            "cp_cold_ms": hybrid_df['cp_cold_ms'].mean(),
            "cp_warm_ms": hybrid_df['cp_warm_ms'].mean(),
            "rl_time": hybrid_df['rl_time'].mean(),
            "cp_cold_time": hybrid_df['cp_cold_time'].mean(),
            "hybrid_time": hybrid_df['hybrid_time'].mean(),
            "gain_vs_cold_ms": hybrid_df['gain_vs_cold_ms'].mean(),
            "speedup_vs_cold": hybrid_df['speedup_vs_cold'].mean()
        }
        hybrid_df = pd.concat([hybrid_df, pd.DataFrame([avg_row])], ignore_index=True)
    else:
        hybrid_df = pd.DataFrame()

    # 3. Save Markdown/LaTeX formats
    with open(output_path, "w") as f:
        f.write("# JSSP Research Results Summary\n\n")
        
        f.write("## 1. Training Phase (Best Val Makespan)\n")
        if not train_summary.empty:
            f.write(train_summary.to_markdown(index=False))
        else:
            f.write("No training data found.\n")
            
        f.write("\n\n## 2. Hybrid Comparison (Taillard Benchmarks)\n")
        if not hybrid_df.empty:
            # Clean display columns
            cols = ["size", "rl_ms", "cp_cold_ms", "cp_warm_ms", "cp_cold_time", "hybrid_time", "speedup_vs_cold"]
            f.write(hybrid_df[cols].to_markdown(index=False))
        else:
            f.write("No hybrid results found.\n")

    print(f"[Success] consolidated report saved to {output_path}")

if __name__ == "__main__":
    # Example usage (usually called by main_train_ppo.py)
    base = os.path.dirname(os.path.dirname(__file__))
    generate_paper_table(
        training_logs_dir=os.path.join(base, "outputs", "logs"),
        hybrid_results_path=os.path.join(base, "eval", "hybrid_experiment_results.csv"),
        output_path=os.path.join(base, "outputs", "final_paper_results.md")
    )
