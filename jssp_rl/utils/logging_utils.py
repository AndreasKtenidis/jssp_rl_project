import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import os

def plot_rl_convergence(episode_makespans, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_makespans) + 1), episode_makespans,
             marker='o', linestyle='-', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Makespan")
    plt.title("RL Convergence over Episodes")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()

def log_epoch(epoch, loss, avg_makespan=None):
    msg = f"📘 Epoch {epoch:3d} | Loss: {loss:.4f}"
    if avg_makespan is not None:
        msg += f" | Avg Makespan: {avg_makespan:.2f}"
    print(msg)

def save_training_metrics(metrics_dict_list, filename="training_metrics_log.csv"):
    df = pd.DataFrame(metrics_dict_list)
    df.to_csv(filename, index=False)
    print(f"📁 Training log saved to {filename}")

def save_training_log_csv(log_data, filename="training_log.csv"):
    df = pd.DataFrame(log_data)
    df.to_csv(filename, index=False)
    print(f"✅ Training log saved to {filename}")


def plot_gantt_chart(job_assignments, save_path=None, title="Gantt Chart ", show_op_index=False):
    """
    Plots a Gantt chart using matplotlib where:
    - Each machine is a row.
    - Each operation is a colored block with job ID label.
    - Time is shown in unit steps (not datetime).
    """
    # Assign colors to jobs
    job_ids = sorted(set(task["job_id"] for task in job_assignments))
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    job_color_map = {job_id: colors[i % len(colors)] for i, job_id in enumerate(job_ids)}

    # Sort by machine, then by start time
    job_assignments = sorted(job_assignments, key=lambda x: (x["machine"], x["start_time"]))

    fig, ax = plt.subplots(figsize=(14, 7))
    machine_count = max(task["machine"] for task in job_assignments) + 1

    yticks = []
    yticklabels = []

    for task in job_assignments:
        job_id = task["job_id"]
        machine = task["machine"]
        start = task["start_time"]
        end = task["end_time"]
        duration = end - start
        label = f"J{job_id + 1}"
        if show_op_index and "operation_index" in task:
            label += f".{task['operation_index']}"

        color = job_color_map[job_id]
        rect = patches.Rectangle((start, machine - 0.4), duration, 0.8, facecolor=color, edgecolor='black')
        ax.add_patch(rect)

        ax.text(start + duration / 2, machine, label, ha='center', va='center', fontsize=7, color='white', weight='bold')

        if machine not in yticks:
            yticks.append(machine)
            yticklabels.append(f"Machine {machine + 1}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time (units)")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    max_time = max(task["end_time"] for task in job_assignments)
    ax.set_xlim(0, max_time + 5)

    for m in yticks:
        ax.axhspan(m - 0.5, m + 0.5, facecolor='lightgray', alpha=0.3, zorder=-1)

    plt.tight_layout()

    if save_path:
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Saved Gantt chart to {save_path}")
    # plt.show()


def plot_cp_vs_rl_comparison(df, save_path="cp_vs_rl_barplot.png"):
    """
    Bar chart comparing CP vs RL on Taillard.
    Supports:
      - df['cp_makespan']                  (required)
      - df['rl_makespan'] or df['rl_makespan_greedy'] (greedy RL; one of these required)
      - df['rl_makespan_best_of_<K>']      (optional; auto-detected)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if "cp_makespan" not in df.columns:
        raise ValueError("DataFrame must contain 'cp_makespan'.")

    # Accept either legacy 'rl_makespan' or new 'rl_makespan_greedy'
    greedy_col = "rl_makespan_greedy" if "rl_makespan_greedy" in df.columns else "rl_makespan"
    if greedy_col not in df.columns:
        raise ValueError("DataFrame must contain 'rl_makespan' or 'rl_makespan_greedy'.")

    # Auto-detect best-of-K column if present (e.g., 'rl_makespan_best_of_10')
    best_cols = [c for c in df.columns if c.startswith("rl_makespan_best_of_")]
    best_col = best_cols[0] if best_cols else None

    indices = df["instance_id"].astype(str).tolist()
    n = len(indices)
    x = np.arange(n)

    # Bar width and offsets for up to 3 series
    width = 0.28 if best_col else 0.4
    offsets = (-width, 0, width) if best_col else (-width/2, width/2)

    plt.figure(figsize=(max(12, n * 0.5), 6))

    # CP bars
    plt.bar(x + (offsets[0] if best_col else offsets[0]), df["cp_makespan"], width=width, label="CP")

    # Greedy RL bars
    plt.bar(x + (offsets[1] if best_col else offsets[1]), df[greedy_col], width=width,
            label="RL (greedy)")

    # Best-of-K RL bars (optional)
    if best_col:
        k = best_col.split("_")[-1]
        plt.bar(x + offsets[2], df[best_col], width=width, label=f"RL (best-of-{k})")

    # Axis formatting
    plt.xticks(x, indices, rotation=45, ha="right")
    plt.xlabel("Instance ID")
    plt.ylabel("Makespan")
    title = "CP vs RL (Greedy"
    title += f" & Best-of-{k}" if best_col else ""
    title += ") on Taillard Instances"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Ensure output dir and save
    outdir = os.path.dirname(save_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Comparison barplot saved to {save_path}")

