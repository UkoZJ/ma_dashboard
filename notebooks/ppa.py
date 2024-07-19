# %% [markdown]
# # Understanding Participation Patterns: Classifying MA-User Behavior

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import MinMaxScaler, Normalizer

root_dir = Path(os.getcwd()).parent
config_path = root_dir.joinpath("config/")
sys.path.append(str(root_dir))

from src import ppa, query_engine
from src.utils import get_config, get_config_filenames

%load_ext autoreload
%autoreload 2

# Get configuration parameters and logger for the current session
config = get_config(path=config_path)
config = get_config(["config.ini"], path=config_path)
config = get_config(get_config_filenames(config["params"]["env"], config_path))

RANDOM_STATE = 0
START_DATE = "2020-01-01"
COLORS = [
    "#669900ff",
    "#003399ff",
    "#2971ffff",
    "#ff6600ff",
    "#ffcc00ff",
    "#bfbfbfff",
]
# %% [markdown]
# Load the precomputed point-pattern analysis dataset.
# %%
qe = query_engine.QueryEngine(config)
qe_ppa = ppa.QueryEngine(config)
qe_ppa.run_ppa(refresh_users_ppa=False)
viz = ppa.Viz(config)

users_stats = pd.read_parquet(config["paths"]["users_ppa_quality"]).set_index("user_id")

# %% [markdown]
# ## Heuristic profiling of users

# We establish threshold values for categorizing users based on their spatial
# dispersion and their participation quality. Users with a high 'adults_ratio'
# value primarily contribute adult-mosquito reports, which typically demand more
# time and effort, while those with a low value engage more frequently with reports
# on bites and breeding sites. The assumption is that users inclined to submit
# adult-mosquito reports are more valuable since they invest greater effort per
# MA-app interaction. Apart from how many adult-reports are submitted by a user,
# it is relevant if those are target-species, measured by the 'quality_target'.

# At this point, it is possible to aggregate quality related measures
# with a harmonic mean since it's useful when dealing with rates or ratios, as
# it tends to give more weight to lower values. There are many cases of 'sporadic'
# users that send only one report where 'quality_target' is 0 but 'adults_ratio'
# is 1; in this case the arithmetic mean would yields 0.5 while a h-mean pulls
# the result toward 0 acting as a logic AND.
#
# There are three levels of user rule-based partitioning, with an increasing
# hierarchical complexity. Level-1 only partition by the number of cluster, Level-2
# adds the partition by user quality, while Level-3 considers all three decision
# variables including spatial dispersion (std_utm).
#
# Here we emphasize the difficulty of using 'quality_reports' for user profiling.
# The discussion revolves around users' accountability in EntoLab's post-processing
# procedures. Despite users submitting high-quality photos, their report-quality
# score often fails to reflect their efforts due to species correlation with
# identification task difficulty. For instance, even if a user consistently submits
# excellent photos of Ae. Japonicus, experts are likely to assign a lower confidence
# level or to disagree more often because of the inherently difficult identification
# task. Assessing user-submitted photo quality remains challenging, with estimations
# focusing at most on photographic quality rather than experts-agreement and
# confidence. As consequence, here we exclude 'quality_report' variable from
# user profiling analysis.

std_lim = 1000
quality_lim = 2.0 / 3
quality_model = "with_additive_cs_acc"

labels, users_global_competence, users_acc = ppa.users_competence(config)

users_stats["adults_ratio"] = users_stats["n_reports"] / users_stats["n_points"]
users_stats["accuracy"] = users_acc["accuracy"]
users_stats["accuracy_chance_adj"] = users_acc["accuracy_chance_adj"]
users_stats["accuracy_chance_adj_pos"] = (
    users_stats["accuracy_chance_adj"].fillna(0).clip(lower=0)
)

match quality_model:
    case "without_cs_acc":
        users_stats["quality"] = hmean(
            users_stats[["adults_ratio", "quality_target"]], axis=1
        )
    case "with_additive_cs_acc":
        users_stats["quality"] = (
            hmean(users_stats[["adults_ratio", "quality_target"]], axis=1)
            + users_stats["accuracy_chance_adj_pos"]
        ) / 2

# Rollover all level of heuristic labels
levels = {1: [None, None], 2: [None, quality_lim], 3: [std_lim, quality_lim]}
for level in [1, 2, 3]:
    std_lim_, quality_lim_ = levels[level]
    users_stats[f"custom_labels_{level}"] = users_stats.apply(
        ppa.user_labels, std_lim=std_lim_, quality_lim=quality_lim_, axis=1
    )

# %% [markdown]
# If we examine the distribution of user accuracy, adjusted for chance, between
# frequent participants (more than 5 reports) and sporadic contributors (fewer
# than 5 reports), we observe that the former group exhibits a more evenly
# distributed range of values. In contrast, sporadic users' accuracy values are
# predominantly 0 or 1 due to the limited data available for accurate evaluation.
# Additionally, it is worth noting that motivated users perform slightly better
# than sporadic ones.

viz.users_quality(users_stats, metric="adjusted accuracy")

# When examining the distribution of User Quality (UQ), motivated users provide,
# on average, higher quality reports compared to sporadic users. Notably, the
# overall number of users considered for UQ is higher than for accuracy evaluation,
# as some users did not supply annotations for adult mosquito reports and thus
# accuracy were not evaluated. However, in computing UQ, when accuracy is missing
# UA takes a value of zero.

viz.users_quality(users_stats, metric="quality")

# %% [markdown]
# ### Overall user-profiling

# First, we present the influx of reports from users classified using heuristic
# labeling, filtered by user origin. We categorize all MA users at a basic
# hierarchical level monthly. It's noted that multi-clustered users appear
# consistently only after the 2021 EU scale-up, with clustered ones increasing
# their presence. As the project matures, sporadic users decrease. The Netherlands'
# 2021-07 communication event primarily mobilized sporadic users. In subsequent
# years, 2022 and 2023, multi-clustered and clustered users notably increase
# presence post-summer surges. Additionally, multi-clustered users maintain high
# presence even in winter due to outstanding commitment. Sparse users' share
# remains constant, reflecting average commitment compared to volatile sporadic
# users.
#
# Notably, the majority of users are retained for up to 1 year. User labeling
# considers entire participation history, cautioning against aggregating user
# distributions below monthly intervals. For robust analysis, quarterly and
# yearly intervals are recommended at expense of local insights. Note that only
# validated reports are considered since the quality score is computed in relation
# to expert validation.

entity = "Total"
level = 1

users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="month"
)
# Overall diagram
viz.level_users_stats(
    users_stats_view,
    level,
    ticker_moltip=6,
    stacked=True,
    is_palette=False,
)

# %% [markdown]
# The second hierarchical level, aggregated monthly, separates users into high-quality
# (upper figure) and low-quality (lower) categories for clearer visualization.
# It's evident that the surge in clustered users after the Spanish 2023 event
# is primarily driven by low-quality users, likely influenced by diverse
# communication channels. Despite their low quality, clustered users, being more
# engaged with the project and favoring spatial sampling, hold greater value
# compared to sporadic users. The latter are not shown in the figure since they
# are not partitioned by the quality measure due to low statistical support.

level = 2
users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="month"
)
viz.level_users_stats(
    users_stats_view.loc[START_DATE:],
    level,
    ticker_moltip=6,
    stacked=True,
    is_palette=False,
    save_data=True,
    title=entity,
)

# %% [markdown]
# The third hierarchical level, aggregated yearly, separates users into high-quality
# (upper figure) and low-quality (lower) categories for clearer visualization.
# This level further distinguishes between clustered and sparse users based on
# the spatial dispersion of their reports (labeled wide and narrow). Sparse users,
# adhering to a wider sampling pattern, contrast with the more balanced approach
# of clustered users. This result is directly dependent on the threshold level
# assigned to the spatial dispersion measure. Through yearly aggregation, it
# becomes apparent that high-quality sparse users decline following the EU scale-up,
# while conversely, low-quality sparse users increase.

level = 3
users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="year"
)
viz.level_users_stats(
    users_stats_view, level, ticker_moltip=None, stacked=False, is_palette=False
)

# %% [markdown]
# ### Spain user-profiling

# The second hierarchical level, aggregated monthly, separates users into high-quality
# (upper figure) and low-quality (lower) categories for clearer visualization.
entity = "Spain"
level = 2
users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="month"
)
# Global picture
viz.level_users_stats(
    users_stats_view.loc[START_DATE:, :],
    level,
    ticker_moltip=3,
    stacked=True,
    is_palette=False,
    save_data=True,
    title=entity,
)

# %% [markdown]
# ### Italy user-profiling

entity = "Italy"
level = 2
users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="month"
)
# Global picture
viz.level_users_stats(
    users_stats_view.loc[START_DATE:, :],
    level,
    ticker_moltip=3,
    stacked=True,
    is_palette=False,
    save_data=True,
    title=entity,
)

# %% [markdown]
# ### Netherlands user-profiling

# The surge in clustered users following the Netherlands' surge event is evidently
# driven primarily by low-quality users. Subsequent to the main communication event,
# the persisting user base comprises predominantly clustered users, with sporadic
# users gradually diminishing as no further communication events engage them.
#
# This case study holds particular interest due to its provision of a clear
# impulsive signal to the system, facilitating a thorough characterization of
# the outcomes. It is noteworthy that post-surge, both clustered and multi-clustered
# users reach their peak share value with a lag of 3 months. While the growth
# and decay rates of multi-clustered users appear consistent, the decay rate of
# clustered users allows for a sustained, long-term presence of this participant
# category even beyond the event. The decay rate of clustered and multi-clustered
# users appears to be approximately equal. A simple grow-decay ODE model would
# confirm this visual observations.

entity = "Netherlands"
level = 2
users_stats_view = qe.ppa_view_inflow(
    users_stats, f"custom_labels_{level}", entity=entity, freq="month"
)
# Global picture
viz.level_users_stats(
    users_stats_view.loc[START_DATE:, :],
    level,
    ticker_moltip=3,
    stacked=True,
    is_palette=False,
    save_data=True,
    title=entity,
)
# Highlight of the surge event
viz.level_users_stats(
    users_stats_view.loc["2021-06-01":"2022-01-01"],
    level,
    ticker_moltip=None,
    stacked=True,
    is_palette=False,
)
# %% [markdown]
# ## Exploratory Data Analysis
#
# ### User Retention
entities = ["Total", "Italy", "Spain", "Netherlands"]
colors = [
    "#669900ff",
    "#003399ff",
    "#ff6600ff",
    "#ffcc00ff",
]
viz.user_retention(qe, entities, freq="quarter", start_date=START_DATE, colors=colors)

# %% [markdown]
# ### MA Active Users
#
# Here, we briefly explore user retention across different 'recency' periods.
# For example, with a recency interval of one year, we count all users who have
# uploaded their most recent report within the past year and mark them as active.
# New users are defined as those who have uploaded at least one report. A user's
# first report serves as the time-marker for their registration in the MA system.
# Note that merely installing the MA app does not qualify someone as a "new user";
# uploading at least one report is required. The percentage of active users is
# calculated as the fraction of active users among new users, aggregated on a
# monthly basis. The following diagram shows the number of active users participating
# in MA as the definition of recency changes.

retention_interval = ["1 year", "6 months", "3 months", "1 months"]
for entity in entities:
    viz.active_users(qe, retention_interval, entity=entity, start_date=START_DATE, colors=colors)

# %% [markdown]
# ## Unsupervised profiling of users

# Dimensional reduction and unsupervised clustering techniques are applied to
# obtain clusters of users, whose behavior depends on geo-location and participation
# quality. The analysis excludes sporadic users, as they contribute insignificantly
# and form a distinct user group. While the number of clusters (n_clusters) is
# discrete, the standard deviation of UTM coordinates (std_utm) is continuous.
# n_clusters are further discretized by grouping users with more than one cluster
# into a multi-cluster category, and a logarithmic transformation is applied to
# std_utm to facilitate and guide dimensional reduction. Level 3 heuristic labeling
# serves as a reference point to validate clusters resulting from the reduction
# process.

users_stats["type_clusters"] = np.where(
    users_stats["n_clusters"] > 1, 2, users_stats["n_clusters"]
)

users_stats["log_std_utm"] = np.log(users_stats["std_utm"])
users_stats["log_std_utm"][users_stats["log_std_utm"] < 0] = 1

users_stats["custom_labels"] = users_stats["custom_labels_3"]

# %% [markdown]
# Optimal results are achieved with the Gower distance measure, which effectively
# handles mixed feature types (categorical and continuous). Results remain consistent
# across UMAP and t-SNE reduction models. It's notable that L2-normalization with
# the Euclidean distance yields slightly different outcomes compared to Cosine
# similarity. Unfortunately, these similarity measures struggle to differentiate
# multi-clustered from clustered users, while Gower distance shows good performance.
# Adding extra features such as n_reports, n_points, and n_random_points doesn't
# yield significant gains.

# Screening the perplexity (n_neighbors) parameter shows robustness in results.
# Comparing clusters with level-3 labels reveals that the quality measure only
# influence sparse users while taking no role for clustered and multi-clustered.
# On the other hand, std_utm influences slightly only clustered users; the fixed
# threshold of std_utm effectively separates this cluster into narrow and wide
# sub-categories. Simplification in heuristic partitioning could be considered
# since std_utm lacks influence over multi-clustered and sparse users.

metric = "gower"

match metric:
    case "gower":
        features = ["log_std_utm", "custom_labels_1", "quality"]
    case _:
        features = ["log_std_utm", "type_clusters", "quality"]

if "custom_labels" not in (features):
    features_ = features + ["custom_labels"]
else:
    features_ = features
users_stats_filt = users_stats.query("custom_labels != 'sporadic'")[features_].fillna(0)

dimred = ppa.DimReduction(
    users_stats_filt,
    metric=metric,
    # n_neighbors=[30, 100, 200, 500],
    n_neighbors=[100],
    feature_transformers={
        "raw": None,
        # "l2": Normalizer(norm="l2")
        # "minmax": MinMaxScaler(feature_range=(0, 1)),
    },
    random_state=RANDOM_STATE,
)

dimred_umap = dimred.screening(feature_cols=features, model_type="umap")
# dimred_tsne = dimred.screening(features, "tsne")

# %% [markdown]
# The number of clusters is selected after visual inspection of the 2D dimensional
# reduced space.

# Select reduction method for further analysis
tf_type = "raw"
dimred = dimred_umap[tf_type]
n_neighbors = 100

# Unsupervised clustering parameters
n_clusters = {"without_cs_acc": 5, "with_additive_cs_acc": 5}

Xft = dimred[0]
Xft_dist = dimred[1]
Xft_dimred = dimred[2].query("neighbors == @n_neighbors").iloc[:, :-2]
dimred_labels = SpectralClustering(
    n_clusters=n_clusters[quality_model], random_state=RANDOM_STATE
).fit_predict(Xft_dimred)
viz.dimred_scatterplot(Xft_dimred, dimred_labels)

# %% [markdown]
# After visual interpretation, we provide the correspondence between dimensional
# reduction clusters and user labels.

viz.labels_boxplot(
    users_stats_filt, dimred_labels, title="Dimensionality Reduction Labels"
)
dimred_labels_map = {
    "without_cs_acc": {
        4: "?",
        0: "?",
        3: "?",
        1: "?",
        2: "?",
    },
    "with_additive_cs_acc": {
        4: "HQ-multi-clustered",
        0: "HQ-clustered narrow",
        1: "LQ-clustered narrow",
        3: "HQ-sparse wide",
        2: "LQ-sparse wide",
    },
}
# %% [markdown]
# For comparison, clustering is repeated but without dimensional reduction. Both
# approaches reach quite a similar result.

cluster_labels = AgglomerativeClustering(
    n_clusters=n_clusters[quality_model],
    linkage="complete",
    metric="precomputed",
).fit_predict(Xft_dist)
viz.dimred_scatterplot(Xft_dimred, cluster_labels)

# %% [markdown]
# As before, after visual interpretation, we provide the correspondence between
# agglomerative clusters and user labels.

viz.labels_boxplot(
    users_stats_filt, cluster_labels, title="Raw data clustering Labels"
)
cluster_labels_map = {
    "without_cs_acc": {
        2: "?",
        1: "?",
        0: "?",
        3: "?",
        4: "?",
    },
    "with_additive_cs_acc": {
        0: "HQ-multi-clustered",
        3: "HQ-clustered narrow",
        1: "LQ-clustered narrow",
        4: "HQ-sparse wide",
        2: "LQ-sparse wide",
    },
}

# %%
users_stats_filt["dimred_labels"] = dimred_labels
users_stats_filt["cluster_labels"] = cluster_labels
users_stats_filt = users_stats_filt.replace(
    {
        "cluster_labels": cluster_labels_map[quality_model],
        "dimred_labels": dimred_labels_map[quality_model],
    }
)

# %% [markdown]
# ### Overall MA-user temporal profiling
# We can now visualize the temporal distribution of user contributions through
# report submissions. For this analysis, we utilize the labels obtained from
# clustering on raw data rather than employing a dimensional reduction procedure.

analyses_type = "cluster_labels"
users_stats["unsupervised_labels"] = users_stats_filt[analyses_type]
order_col = eval(f"list({analyses_type}_map[quality_model].values())")

users_stats["unsupervised_labels"] = users_stats["unsupervised_labels"].fillna(
    "sporadic"
)
order_col_ = order_col + ["sporadic"]
entity = "Total"

users_stats_view = qe.ppa_view_inflow(
    users_stats, "unsupervised_labels", entity=entity, freq="month"
)
color = [
    "#669900ff",
    "#003399ff",
    "#689aff",
    "#ff6600ff",
    "#ffcc00",
    "#bfbfbfff",
]
viz.users_stats(users_stats_view, order_col_, ticker_moltip=6, colors=color)

# %% [markdown]
# ### Spanish users temporal profiling
# Examining the exceptional surges in Spain (2023) and the Netherlands (2021),
# it becomes apparent that sporadic users are initially involved, but their
# contributions gradually diminish, giving way to the increased participation
# of multi-clustered and clustered users.

entity = "Spain"
users_stats_view = qe.ppa_view_inflow(
    users_stats, "unsupervised_labels", entity=entity, freq="month"
)
viz.users_stats(users_stats_view, order_col_, ticker_moltip=6, colors=color)

# %% [markdown]
# ### Netherlands users temporal profiling
entity = "Netherlands"
users_stats_view = qe.ppa_view_inflow(
    users_stats, "unsupervised_labels", entity=entity, freq="month"
)
viz.users_stats(
    users_stats_view.loc[START_DATE:, :], order_col_, ticker_moltip=6, colors=color
)

# %% [markdown]
# ### Italian users temporal profiling
entity = "Italy"
users_stats_view = qe.ppa_view_inflow(
    users_stats, "unsupervised_labels", entity=entity, freq="month"
)
viz.users_stats(
    users_stats_view.loc[START_DATE:, :], order_col_, ticker_moltip=6, colors=color
)

