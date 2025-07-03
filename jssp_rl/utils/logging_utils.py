import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

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
        gantt_data.append({
            "Job": f"Job {task['job_id'] + 1}",
            "Machine": f"Machine {task['machine'] + 1}",
            "Start": task["start_time"],
            "End": task["end_time"],
            "Duration": task["end_time"] - task["start_time"],
        })

    df = pd.DataFrame(gantt_data)

    fig = px.timeline(df,
                      x_start="Start", x_end="End",
                      y="Machine", color="Job",
                      title=title)

    fig.update_yaxes(categoryorder="category descending")

    if save_path:
        fig.write_html(save_path)
        print(f"📊 Gantt chart saved to {save_path}")

    fig.show()

