import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as mcolors



def plot_rl_convergence(episode_makespans, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_makespans) + 1), episode_makespans, marker='o', linestyle='-', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Makespan")
    plt.title("RL Convergence over Episodes")
    plt.grid(True)
    if save_path:
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

        # Write job ID inside the rectangle
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

    # Add gray background for each machine row
    for m in yticks:
        ax.axhspan(m - 0.5, m + 0.5, facecolor='lightgray', alpha=0.3, zorder=-1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved Gantt chart to {save_path}")
    # plt.show()


def plot_cp_vs_rl_comparison(df, save_path="cp_vs_rl_barplot.png"):
    
    plt.figure(figsize=(12, 6))
    indices = df["instance_id"].astype(str)
    x = range(len(indices))

    plt.bar(x, df["cp_makespan"], width=0.4, label="CP", align='center')
    plt.bar([i + 0.4 for i in x], df["rl_makespan"], width=0.4, label="RL", align='center')

    plt.xticks([i + 0.2 for i in x], indices, rotation=45)
    plt.xlabel("Instance ID")
    plt.ylabel("Makespan")
    plt.title("Comparison of CP vs RL on Taillard Instances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Comparison barplot saved to {save_path}")