import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pandas import Timestamp, to_timedelta

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



def plot_gantt_chart(job_assignments, save_path=None, title="Gantt Chart"):
    """
    Plot an interactive Gantt chart from job assignment list.
    Each job has a start_time, end_time, machine, and job_id.
    """
    gantt_data = []

    for task in job_assignments:
        start = task["start_time"]
        end = task["end_time"]

        # Convert floats to timestamps if necessary
        if isinstance(start, (int, float)):
            start = Timestamp("2023-01-01") + to_timedelta(start, unit='s')
        if isinstance(end, (int, float)):
            end = Timestamp("2023-01-01") + to_timedelta(end, unit='s')

        gantt_data.append({
            "Job": f"Job {task['job_id'] + 1}",
            "Machine": f"Machine {task['machine'] + 1}",
            "Start": start,
            "End": end,
            "Duration": (end - start).total_seconds(),
        })

    df = pd.DataFrame(gantt_data)

    fig = px.timeline(df,
                      x_start="Start", x_end="End",
                      y="Machine", color="Job",
                      title=title)

    fig.update_yaxes(categoryorder="category descending")

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
        print(f"📊 Gantt chart saved to {save_path}")

    fig.show()

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