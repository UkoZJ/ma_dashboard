# %%
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS


plt.style.use("ggplot")
colors = ["#00ADB5", "#FF5376", "#724BE5", "#FDB62F"]
plt.rcParams.update({"font.size": 15})


def plot_clusters(algorithm, _df, _min_samples, _eps=""):
    unique_labels = set(_df.labels)
    for k, col in zip(unique_labels, colors[0 : len(unique_labels)]):
        # Use black color for noise
        if k == -1:
            col = "k"
        # Use different color per cluster and add labels
        plt.plot(
            _df.loc[_df.labels == k].x,
            _df.loc[_df.labels == k].y,
            "o",
            color=col,
            markeredgecolor="k",
            markersize=15,
            label=(f"Cluster {k+1}" if k != -1 else "Noise")
            + f" ({_df.loc[_df.labels == k].shape[0]})",
        )
    # Add legend and title
    plt.legend(loc="upper right")
    plt.title(
        f"{algorithm}: "
        + (f"eps={_eps}, " if _eps != "" else _eps)
        + f"min_samples={_min_samples}"
    )
    plt.show()


def reachability_plot(_df, model):
    # Get reachability distances and cluster labels
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]
    unique_labels = set(labels)
    space = np.arange(len(_df))
    # Generate reachability plot using different color per cluster
    for k, col in zip(unique_labels, colors):
        xk = space[labels == k]
        rk = reachability[labels == k]
        plt.plot(xk, rk, col)
        plt.fill_between(xk, rk, color=col, alpha=0.5)
    # Ordering in x-axis
    plt.xticks(space, _df.index[model.ordering_], fontsize=10)
    # Plot outliers
    plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    # Add y-label and title
    plt.ylabel("Reachability Distance")
    plt.title("Reachability Plot")
    plt.show()


# Generate data
centers = [[1, 1], [-2, -4], [5, -7]]
data = make_blobs(
    n_samples=[30, 20, 10],
    centers=centers,
    cluster_std=[0.8, 1, 1.5],
    random_state=0,
)[0]
data = StandardScaler().fit_transform(data)
df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1]))

# Define parameters
eps = 0.5
min_samples = 5
metric = "euclidean"

# Perform DBSCAN clustering and visualize results
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(df)
df["labels"] = dbscan.labels_
plot_clusters("DBSCAN", df, str(min_samples), str(eps))

# Perform OPTICS clustering and visualize results
optics = OPTICS(max_eps=eps * 2, min_samples=min_samples, metric=metric).fit(df)
df["labels"] = optics.labels_
plot_clusters("OPTICS", df, str(min_samples))
reachability_plot(df, optics)

# %%

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Example data
centers = [[1, 1], [-2, -4], [5, -7]]
data = make_blobs(
    n_samples=[30, 20, 10],
    centers=centers,
    cluster_std=[0.8, 1, 1.5],
    random_state=0,
)[0]
data = StandardScaler().fit_transform(data)
df = pd.DataFrame(dict(x=data[:, 0], y=data[:, 1]))

# DBSCAN clustering
# db = DBSCAN(eps=0.5, min_samples=10).fit(df)
db = OPTICS(min_samples=5).fit(df)
labels = db.labels_

# Plotting results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering")
plt.show()

ds = pd.Series(db.labels_)
labels_count = ds.value_counts()
if -1 in labels_count.index:
    if len(labels_count) > 1:
        labels_count = labels_count.drop(-1)
        n_clusters = len(labels_count)
    else:
        n_clusters = 0
else:
    n_clusters = len(labels_count)
print(f"Number of clusters: {n_clusters}")

# Identifying noise
num_noise_points = sum(labels == -1)
print(f"Number of noise points: {num_noise_points}")
# %%
