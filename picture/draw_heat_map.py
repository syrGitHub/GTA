'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 这里是创建一个数据
yyy = ["Speed", "Temperature", "Density", "Pressure",
              "Sigma-B", "ICME", "Coronal hole information"]
xxx = ["Speed", "Temperature", "Density", "Pressure",
              "Sigma-B", "ICME", "Coronal hole information"]

harvest_1 = np.array([[1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.7],
                    [0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.6],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.5, 1.0, 0.0, 0.8, 0.0],
                    [0.0, 0.8, 0.0, 0.6, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.7],
                    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4, 1.0]])
harvest_2 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.8],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.8, 1.0, 0.0, 0.5, 0.0],
                    [0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 0.8],
                    [0.0, 0.0, 0.7, 0.0, 0.0, 1.0, 0.8],
                    [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 1.0]])
harvest_3 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.6],
                    [0.7, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 1.0]])
harvest_4 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 1.0]])

print("okkk")

# 这里是创建一个画布
fig = plt.figure(1, figsize=(12, 9.5))
# im = ax.imshow(harvest)

ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(harvest_1, cmap='Reds')
# 这里是修改标签
# We want to show all ticks...
ax1.set_xticks(np.arange(len(xxx)))
ax1.set_yticks(np.arange(len(yyy)))
# ... and label them with the respective list entries
ax1.set_xticklabels(xxx)
ax1.set_yticklabels(yyy)
ax1.set_title('Horizon=24h')

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


ax2 = fig.add_subplot(2, 2, 2)
sns.heatmap(harvest_2, cmap='Reds')
# 这里是修改标签
# We want to show all ticks...

ax2.set_xticks(np.arange(len(xxx)))
ax2.set_yticks(np.arange(len(yyy)))
# ... and label them with the respective list entries
ax2.set_xticklabels(xxx)
ax2.set_yticklabels(yyy)
ax2.set_title('Horizon=48h')

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


ax3 = fig.add_subplot(2, 2, 3)
sns.heatmap(harvest_3, cmap='GnBu')
# 这里是修改标签
# We want to show all ticks...
ax3.set_xticks(np.arange(len(xxx)))
ax3.set_yticks(np.arange(len(yyy)))
# ... and label them with the respective list entries
ax3.set_xticklabels(xxx)
ax3.set_yticklabels(yyy)
ax3.set_title('Horizon=72h')

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


ax4 = fig.add_subplot(2, 2, 4)
sns.heatmap(harvest_4, cmap='GnBu')

# 这里是修改标签
# We want to show all ticks...
ax4.set_xticks(np.arange(len(xxx)))
ax4.set_yticks(np.arange(len(yyy)))
# ... and label them with the respective list entries
ax4.set_xticklabels(xxx)
ax4.set_yticklabels(yyy)
ax4.set_title('Horizon=96h')

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax4.get_yticklabels(), rotation=360, horizontalalignment='right')
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


# ax1.set_title("Harvest of Graph Attention Module")
fig.tight_layout()
# plt.colorbar(im)
plt.savefig('/home/sunyanru19s/pytorch/GDN-main/picture/24h.png', dpi=300)
plt.show()
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ax4.set_xticks(np.arange(len(xxx)))
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

vegetables = ["Speed", "Temperature", "Density", "Pressure",
              "Sigma-B", "ICME", "Coronal hole information"]
farmers = ["Speed", "Temperature", "Density", "Pressure",
              "Sigma-B", "ICME", "Coronal hole information"]

harvest_1 = np.array([[1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.7],
                    [0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.6],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.5, 1.0, 0.0, 0.8, 0.0],
                    [0.0, 0.8, 0.0, 0.6, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.7],
                    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4, 1.0]])
harvest_2 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.8],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.8, 1.0, 0.0, 0.5, 0.0],
                    [0.0, 0.6, 0.0, 0.0, 1.0, 0.0, 0.8],
                    [0.0, 0.0, 0.7, 0.0, 0.0, 1.0, 0.8],
                    [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 1.0]])
harvest_3 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.6],
                    [0.7, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 1.0]])
harvest_4 = np.array([[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8],
                    [0.0, 0.0, 1.0, 0.9, 0.0, 0.7, 0.0],
                    [0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.8],
                    [0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 1.0]])



fig = plt.figure(1, figsize=(12, 9.5))
# im = ax.imshow(harvest)

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Horizon=24h')
im, cbar = heatmap(harvest_1, vegetables, farmers, cmap="Reds")

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Horizon=48h')
im, cbar = heatmap(harvest_2, vegetables, farmers, cmap="Reds")

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Horizon=72h')
im, cbar = heatmap(harvest_3, vegetables, farmers, cmap="YlGn")

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Horizon=96h')
im, cbar = heatmap(harvest_4, vegetables, farmers, cmap="YlGn")


fig.tight_layout()
plt.savefig('/home/sunyanru19s/pytorch/GDN-main/picture/24h.png', dpi=300)
plt.show()
