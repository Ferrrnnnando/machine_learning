import matplotlib.pyplot as plt
import seaborn as sns


"""Plot histogram or box diagram on numerical data columns
"""


def plot_data(data, plot_type, grid_size, fig_size, y=None):
    fig = plt.figure(figsize=fig_size)
    column_names = data.select_dtypes(exclude="object").columns

    for i, column_name in enumerate(column_names):
        fig.add_subplot(grid_size[0], grid_size[1], i + 1)
        if plot_type == "hist":
            plot = sns.histplot(data[column_name], kde=True, color="red")
        elif plot_type == "boxplot":
            plot = sns.boxplot(y=data[column_name], x=y, color="red")
        else:
            raise ValueError(
                "Input value for the parameter 'plot_type' should be 'hist' or 'boxplot'."
            )
        plot.set_xlabel(column_name, fontsize=16)
    plt.tight_layout()
