import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter


def delete_existing_file(file_path):
    """Check if a file exists at the specified path. If it does, delete it.

    Args:
        file_path: str
            The path to the file to be deleted.
    """

    if os.path.exists(file_path):
        os.remove(file_path)


def plot_and_save_correlation_heatmaps(corr_values_nodes, corr_p_nodes, corr_values_layers, corr_p_layers, algo_names,
                                       nodes_title, layers_title, suptitle, short_title, direction, save_path=None):
    """Plot two side-by-side heatmaps for correlation matrices. Optionally, check if the file located at save_path existing, delete it if yes, and save the new figure.

    Args:
        corr_matrix_nodes: pandas.DataFrame
            A DataFrame with correlation values for nodes.
        corr_matrix_layers: pandas.DataFrame
            A DataFrame with correlation values for layers.
        nodes_title: str
            Title for the nodes heatmap.
        layers_title: str
            Title for the layers heatmap.
        suptitle: str
            Main title for the entire figure.
        save_path: str
            Path to save the figure. If None, the figure is not saved. Deletes existing file if present.
        direction: str, optional
            The direction of the subplots. Must be either 'horizontal' or 'vertical'.
    """

    corr_matrix_nodes = pd.DataFrame(corr_values_nodes, columns=algo_names, index=algo_names)
    corr_matrix_layers = pd.DataFrame(corr_values_layers, columns=algo_names, index=algo_names)

    sns.set_theme(style='white', font='Helvetica', font_scale=1.4)

    # Create figure with subplots
    match direction:
        case 'horizontal':
            plt.figure(figsize=(16, 8))
            ax1 = plt.subplot(1, 2, 1)
            rotate = 45
        case 'vertical':
            plt.figure(figsize=(8, 16))
            ax1 = plt.subplot(2, 1, 1)
            rotate = 0
        case _:
            raise ValueError("Invalid direction. Must be 'horizontal' or 'vertical'.")

    heatmap1 = sns.heatmap(corr_matrix_nodes, annot=corr_p_nodes, fmt='.2f', cmap='Greens', linewidths=.5,
                           cbar_kws={'shrink': .7, 'format': FormatStrFormatter('%.2f')}, square=True)
    heatmap1.set_xticklabels(heatmap1.get_xticklabels(), rotation=rotate)
    heatmap1.set_yticklabels(heatmap1.get_yticklabels(), rotation=rotate)
    ax1.set_title(nodes_title, fontsize=20, pad=12)

    match direction:
        case 'horizontal':
            ax2 = plt.subplot(1, 2, 2)
        case 'vertical':
            ax2 = plt.subplot(2, 1, 2)
        case _:
            raise ValueError("Invalid direction. Must be 'horizontal' or 'vertical'.")

    heatmap2 = sns.heatmap(corr_matrix_layers, annot=corr_p_layers, fmt='.2f', cmap='Blues', linewidths=.5,
                           cbar_kws={'shrink': .7, 'format': FormatStrFormatter('%.2f')}, square=True)
    heatmap2.set_xticklabels(heatmap2.get_xticklabels(), rotation=rotate)
    heatmap2.set_yticklabels(heatmap2.get_yticklabels(), rotation=rotate)
    ax2.set_title(layers_title, fontsize=20, pad=12)

    match direction:
        case 'horizontal':
            plt.suptitle(suptitle, fontsize=20, y=0.93)
            plt.tight_layout(rect=[0, 0, 1, 1])
        case 'vertical':
            plt.suptitle(short_title, fontsize=22, y=0.92)
            plt.subplots_adjust(wspace=0, hspace=0)
        case _:
            raise ValueError("Invalid direction. Must be 'horizontal' or 'vertical'.")

    if save_path:
        delete_existing_file(save_path)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()


def plot_and_save_diff_percentage_lineplot(data, colours_group, xlabel, ylabel, title, save_path=None):
    """Plot a line graph for diff percentages across multiple methods, with options to save it.

    Args:
        data: pandas.DataFrame
            A DataFrame where each column represents a different method's data points, and includes an 'x-axis' column.
        xlabel: str
            Label for the x-axis, which represents the measurement or category.
        ylabel: str
            Label for the y-axis, which shows the percentage of difference.
        title: str
            The title of the graph, describing the data being visualized.
        save_path: str, optional
            The file path where the plot image will be saved. If not provided, the plot is not saved. If provided, checks for existing file at the path and deletes it before saving the new plot.
    """

    # Melting the DataFrame for seaborn
    data_melted = data.melt(id_vars=['x-axis'], var_name='Method', value_name='Difference Percentage')

    sns.set_theme(style='white', font='Helvetica', font_scale=1.3)
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=data_melted, x='x-axis', y='Difference Percentage', hue='Method',
                 palette=colours_group, linewidth=2)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.fill_between(data['x-axis'], 0, data['LayerPlexRank'], color='bisque')

    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.axvline(x=len(data['x-axis']), color='k', linestyle='--')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=22)
    plt.grid(True)
    plt.legend(fontsize='20', loc=(0.1, 0.4))

    if save_path:
        delete_existing_file(save_path)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()


def plot_and_save_para_sensitivity(data_list, xlabel, ylabel, title, save_path=None):
    """Plot a line graph for sensitivity analysis across multiple parameters, with options to save it.

    Args:
        data_list: list
            A list of dictionaries where each dictionary represents a different method's data points.
        xlabel: str
            Label for the x-axis, which represents the parameter being varied.
        ylabel: str
            Label for the y-axis, which shows the rank of the method.
        title: str
            The title of the graph, describing the data being visualized.
        save_path: str, optional
            The file path where the plot image will be saved. If not provided, the plot is not saved. If provided, checks for existing file at the path and deletes it before saving the new plot.
    """

    sens_data = pd.DataFrame(data_list)

    sns.set_theme(style='white', font='Helvetica', font_scale=1.3)
    plt.figure(figsize=(12, 8))
    plt.gca().invert_yaxis()

    sns.lineplot(data=sens_data, x='gamma', y='rank', hue='node_id', palette='tab10', legend=False)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=22)

    if save_path:
        delete_existing_file(save_path)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()
