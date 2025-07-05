# jssp_rl/utils/gantt_utils.py

import plotly.figure_factory as ff
import pandas as pd
import os

def plot_gantt_chart(job_assignments, save_path=None):
    colors = {}
    df = []

    for job, machine, start, end in job_assignments:
        label = f"Job {job}"
        df.append(dict(Task=f"Machine {machine}", Start=start, Finish=end, Resource=label))
        colors[label] = f"rgb({job * 17 % 255}, {job * 47 % 255}, {job * 97 % 255})"

    fig = ff.create_gantt(df, index_col='Resource', show_colorbar=True, group_tasks=True, colors=colors)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    return fig
