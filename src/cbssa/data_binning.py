import numpy as np
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot as plt


def get(
        buckets: list[float],
        col1: pd.DataFrame | np.ndarray,
        col2: pd.DataFrame | np.ndarray,
        print_table: bool = False
) -> pd.DataFrame:

    data_dic = {}

    idx_label = []
    count = []
    avg1 = []
    avg2 = []
    stderr2 = []
    i = None

    for i in range(len(buckets) - 1):
        idx_label.append("[%s,%s)" % (buckets[i], buckets[i + 1]))
        count.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].count())
        avg1.append(col1[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        avg2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].mean())
        stderr2.append(col2[(col1 >= buckets[i]) & (col1 < buckets[i + 1])].sem() * 2)

    if i:
        idx_label[-1] = f"[{buckets[i]}, {buckets[i + 1]}]"

    data_dic["Bins"] = idx_label
    data_dic["Count"] = count
    data_dic[f"Avg {col1.name}"] = avg1
    data_dic[f"Avg {col2.name}"] = avg2
    data_dic[f"Stderr {col2.name}"] = stderr2

    order_list = ["Bins", "Count", f"Avg {col1.name}", f"Avg {col2.name}", f"Stderr {col2.name}"]
    summary_table = pd.DataFrame(data=data_dic)[order_list]

    if print_table:
        print(summary_table)

    return summary_table


def graph(
        binned_stats: pd.DataFrame,
        show_graph: bool = False,
) -> matplotlib.figure.Figure:

    col_name = list(binned_stats.columns.values)
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(
        binned_stats[col_name[2]], binned_stats[col_name[3]], yerr=binned_stats[col_name[4]], fmt=".", capsize=5
    )
    if show_graph:
        plt.show()

    return fig


def graph_with_prediction(
        binned_stats: pd.DataFrame,
        line_x: pd.Series,
        line_y: pd.Series,
        line_style: str,
        line_x2: pd.Series = None,
        line_y2: pd.Series = None,
        line_style_2: str = None,
        show_graph: bool = False
) -> matplotlib.figure.Figure:

    col_name = list(binned_stats.columns.values)
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(
        binned_stats[col_name[2]], binned_stats[col_name[3]], yerr=binned_stats[col_name[4]], fmt=".", capsize=5
    )

    if line_x2 is not None:
        plt.plot(line_x, line_y, line_style, line_x2, line_y2, line_style_2)

    else:
        plt.plot(line_x, line_y, line_style)

    plt.xlabel("distance")
    plt.ylabel("make")

    if show_graph:
        plt.show()

    return fig
