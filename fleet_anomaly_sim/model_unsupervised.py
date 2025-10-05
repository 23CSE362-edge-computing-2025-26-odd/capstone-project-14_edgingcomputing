import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


def analyze_fleet(csv_file, method="single", plot=True, thrcc=0.9):
    """
    Fleet-based anomaly detection using hierarchical clustering
    (Hendrickx et al., 2020).
    """
    # load data
    df = pd.read_csv(csv_file)
    data = df["vibration"].values
    n_machines = len(data)

    print(f"Loaded {n_machines} machines")

    # normalization (min-max)
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min == 0:
        data_normalized = np.zeros_like(data)
    else:
        data_normalized = (data - data_min) / (data_max - data_min)

    # pairwise distance matrix
    def calculate_distances(values):
        n = len(values)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(values[i] - values[j])
                distances[i, j] = distances[j, i] = dist
        return distances

    distance_matrix = calculate_distances(data_normalized)

    # convert to condensed form for scipy
    condensed_dist = squareform(distance_matrix)

    # hierarchical clustering
    Z = linkage(condensed_dist, method=method)

    # cut the dendrogram
    max_distance = np.max(Z[:, 2])
    cut_distance = thrcc * max_distance
    clusters = fcluster(Z, t=cut_distance, criterion='distance')

    # anomaly_score(machine) = 1 - (cluster_size / total_machines)
    anomaly_scores = np.zeros(n_machines)
    for i in range(n_machines):
        cluster_id = clusters[i]
        cluster_size = np.sum(clusters == cluster_id)
        anomaly_scores[i] = 1 - (cluster_size / n_machines)

    # thresholding
    anomaly_threshold = 2/3
    anomalous_machines = np.where(anomaly_scores > anomaly_threshold)[0].tolist()
    healthy_machines = np.where(anomaly_scores <= anomaly_threshold)[0].tolist()

    print(f"Clusters found: {len(np.unique(clusters))}")
    print(f"Healthy machines: {healthy_machines}")
    print(f"Anomalous machines: {anomalous_machines}")

    results = {
        "clusters": clusters,
        "anomaly_scores": anomaly_scores,
        "healthy_machines": healthy_machines,
        "anomalous_machines": anomalous_machines,
        "linkage_matrix": Z,
        "original_data": data,
        "normalized_data": data_normalized,
    }

    if plot:
        create_plots(results, n_machines)

    return results


def create_plots(results, n_machines):
    """Visualization similar to paper (signal view, dissimilarity, dendrogram, anomaly scores)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    clusters = results["clusters"]
    anomaly_scores = results["anomaly_scores"]
    data = results["original_data"]

    # dendrogram
    dendrogram(results["linkage_matrix"], ax=axes[0, 0])
    axes[0, 0].set_title("Hierarchical Clustering Dendrogram")
    axes[0, 0].set_xlabel("Machine Index")
    axes[0, 0].set_ylabel("Distance")

    # raw data scatter
    colors = ["green" if idx in results["healthy_machines"] else "red" for idx in range(n_machines)]
    axes[0, 1].scatter(range(n_machines), data, c=colors, s=50)
    axes[0, 1].set_title("Fleet Data (Green=Healthy, Red=Anomalous)")
    axes[0, 1].set_xlabel("Machine Index")
    axes[0, 1].set_ylabel("Vibration")
    axes[0, 1].grid()

    # anomaly scores
    bar_colors = ["red" if score > 2/3 else "green" for score in anomaly_scores]
    axes[1, 0].bar(range(n_machines), anomaly_scores, color=bar_colors)
    axes[1, 0].axhline(y=2/3, color="black", linestyle="--", label="Anomaly Threshold (2/3)")
    axes[1, 0].set_title("Anomaly Scores")
    axes[1, 0].set_xlabel("Machine Index")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # cluster sizes
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    axes[1, 1].bar(unique_clusters, counts, color="skyblue", edgecolor="black")
    axes[1, 1].set_title("Cluster Sizes")
    axes[1, 1].set_xlabel("Cluster ID")
    axes[1, 1].set_ylabel("Size")
    axes[1, 1].grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = analyze_fleet("edge/Data/simulated_vibration_edge2.csv",
                            method="single", plot=True, thrcc=0.2)
