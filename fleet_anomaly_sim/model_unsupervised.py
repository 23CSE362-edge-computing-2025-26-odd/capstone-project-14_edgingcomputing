import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage, fcluster


def analyze_fleet(csv_file, method="ward", plot=True, distance_threshold=0.3):
    """
    analyze fleet data
    """
    # load data
    df = pd.read_csv(csv_file)
    data = df["vibration"].values
    n_machines = len(data)
    
    print(f"Loaded {n_machines} machines")
    print(f"Data range: {np.min(data):.4f} to {np.max(data):.4f}")
    
    # min-max scaling
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min == 0:
        data_normalized = np.zeros_like(data)
    else:
        data_normalized = (data - data_min) / (data_max - data_min)
    
    # pairwise distance calculation
    def calculate_distances(values):
        n = len(values)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = abs(values[i] - values[j])
                distances[i, j] = distances[j, i] = dist
        return distances
    
    # distance matrix
    distance_matrix = calculate_distances(data_normalized)
    
    # condensed form for scipy
    condensed_dist = []
    for i in range(n_machines):
        for j in range(i+1, n_machines):
            condensed_dist.append(distance_matrix[i, j])
    condensed_dist = np.array(condensed_dist)
    
    # hierarchical clustering
    try:
        Z = linkage(condensed_dist, method=method)
        print(f"Clustering successful with {method} method")
    except Exception as e:
        print(f"Error in clustering: {e}")
        # Fallback: simple distance-based clustering
        return simple_distance_clustering(data, data_normalized, plot)
    
    # determine clusters by dendrogram
    max_distance = np.max(Z[:, 2])
    cut_distance = distance_threshold * max_distance
    clusters = fcluster(Z, t=cut_distance, criterion='distance')
    
    print(f"Found {len(np.unique(clusters))} clusters")
    for i, cluster_id in enumerate(np.unique(clusters)):
        cluster_size = np.sum(clusters == cluster_id)
        print(f"Cluster {cluster_id}: {cluster_size} machines")
    
    # healthy cluster (largest cluster)
    cluster_counts = np.bincount(clusters)
    # skip index 0 since cluster IDs start from 1
    if len(cluster_counts) > 1:
        healthy_cluster_id = np.argmax(cluster_counts[1:]) + 1
    else:
        healthy_cluster_id = 1
    
    healthy_machines = np.where(clusters == healthy_cluster_id)[0]
    anomalous_machines = np.where(clusters != healthy_cluster_id)[0]
    
    print(f"Healthy cluster: {healthy_cluster_id} ({len(healthy_machines)} machines)")
    print(f"Anomalous machines: {len(anomalous_machines)}")
    
    # healthy centroid
    healthy_data = data_normalized[healthy_machines]
    healthy_centroid = np.mean(healthy_data)
    
    # distances to healthy centroid
    distances_to_centroid = np.abs(data_normalized - healthy_centroid)
    
    # anomaly scores
    anomaly_scores = np.zeros(n_machines)
    for i in range(n_machines):
        if clusters[i] == healthy_cluster_id:
            anomaly_scores[i] = 0.0  # healthy
        else:
            # score based on distance to healthy centroid
            max_dist = np.max(distances_to_centroid)
            if max_dist > 0:
                anomaly_scores[i] = distances_to_centroid[i] / max_dist
            else:
                anomaly_scores[i] = 1.0
    
    results = {
        "clusters": clusters,
        "anomaly_scores": anomaly_scores,
        "distances": distances_to_centroid,
        "healthy_machines": healthy_machines,
        "anomalous_machines": anomalous_machines,
        "healthy_centroid": healthy_centroid,
        "linkage_matrix": Z,
        "original_data": data,
        "normalized_data": data_normalized
    }
    
    if plot:
        create_plots(results, n_machines, healthy_cluster_id)
    
    return results


def simple_distance_clustering(data, data_normalized, plot):
    """Fallback clustering method using simple distance threshold"""
    n_machines = len(data)
    
    # median and standard deviation
    median_val = np.median(data_normalized)
    std_val = np.std(data_normalized)
    
    # threshold for anomaly (2 standard deviations)
    threshold = 2.0 * std_val
    
    # classify machines
    distances_to_median = np.abs(data_normalized - median_val)
    clusters = np.ones(n_machines, dtype=int)
    anomaly_scores = np.zeros(n_machines)
    
    for i in range(n_machines):
        if distances_to_median[i] > threshold:
            clusters[i] = 2  # anomalous cluster
            anomaly_scores[i] = min(distances_to_median[i] / (3 * std_val), 1.0)
    
    healthy_machines = np.where(clusters == 1)[0]
    anomalous_machines = np.where(clusters == 2)[0]
    
    print(f"Fallback method: {len(healthy_machines)} healthy, {len(anomalous_machines)} anomalous")
    
    results = {
        "clusters": clusters,
        "anomaly_scores": anomaly_scores,
        "distances": distances_to_median,
        "healthy_machines": healthy_machines,
        "anomalous_machines": anomalous_machines,
        "healthy_centroid": median_val,
        "linkage_matrix": None,
        "original_data": data,
        "normalized_data": data_normalized
    }
    
    if plot:
        create_plots(results, n_machines, 1)
    
    return results


def create_plots(results, n_machines, healthy_cluster_id):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    clusters = results["clusters"]
    anomaly_scores = results["anomaly_scores"]
    data = results["original_data"]
    healthy_centroid = results["healthy_centroid"]
    distances = results["distances"]
    
    # dendrogram
    if results["linkage_matrix"] is not None:
        try:
            dendrogram(results["linkage_matrix"], ax=axes[0, 0])
            axes[0, 0].set_title("Hierarchical Clustering Dendrogram")
            axes[0, 0].set_xlabel("Machine Index")
            axes[0, 0].set_ylabel("Distance")
        except:
            axes[0, 0].text(0.5, 0.5, "Dendrogram not available", 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("Dendrogram (Not Available)")
    else:
        axes[0, 0].text(0.5, 0.5, "Simple distance-based clustering used", 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("Clustering Method")
    
    # scatter plot
    colors = []
    for i in range(n_machines):
        if clusters[i] == healthy_cluster_id:
            colors.append('green')
        else:
            colors.append('red')
    
    axes[0, 1].scatter(range(n_machines), data, c=colors, alpha=0.7, s=50)
    
    # centroid line (convert back to original scale)
    data_min, data_max = np.min(data), np.max(data)
    centroid_original = healthy_centroid * (data_max - data_min) + data_min
    axes[0, 1].axhline(y=centroid_original, color="blue", linestyle="--", 
                      alpha=0.7, label=f"Healthy Centroid ({centroid_original:.4f})")
    
    axes[0, 1].set_title(f"Fleet Data (Green=Healthy, Red=Anomalous)")
    axes[0, 1].set_xlabel("Machine Index")
    axes[0, 1].set_ylabel("Vibration Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # anomaly scores
    bar_colors = ['red' if score > 0.1 else 'green' for score in anomaly_scores]
    axes[1, 0].bar(range(n_machines), anomaly_scores, color=bar_colors, alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, 
                      label='High Anomaly Threshold')
    axes[1, 0].set_title("Anomaly Scores")
    axes[1, 0].set_xlabel("Machine Index")
    axes[1, 0].set_ylabel("Anomaly Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # distance distribution
    axes[1, 1].hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(distances), color='red', linestyle='--', 
                      label=f'Mean Distance ({np.mean(distances):.4f})')
    axes[1, 1].set_title("Distribution of Distances to Healthy Centroid")
    axes[1, 1].set_xlabel("Distance")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # statistics
    print(f"Total machines: {n_machines}")
    print(f"Healthy machines: {len(results['healthy_machines'])}")
    print(f"Anomalous machines: {len(results['anomalous_machines'])}")
    print(f"Healthy centroid: {centroid_original:.6f}\n\nThe following are the clusters identified, that are anomalous:")
    
    # print clusters with ranges and values
    unique_clusters = np.unique(clusters)
    for count,cluster_id in enumerate(unique_clusters):
        if count == len(unique_clusters)-1:
            break
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_values = data[cluster_indices]
        cluster_range = (np.min(cluster_values), np.max(cluster_values))
        print(f"Cluster {cluster_id} ({cluster_range[0]:.4f} - {cluster_range[1]:.4f}): {cluster_values.tolist()}\n")

    
    # show top 5 anomaly scores
    top_anomaly_indices = np.argsort(anomaly_scores)[-5:][::-1]
    print(f"\nTop 5 anomaly scores:")
    for idx in top_anomaly_indices:
        print(f"Machine {idx}: score={anomaly_scores[idx]:.4f}, value={data[idx]:.6f}")


# using it on a subset of the generated data, wrapped in main so that it doesn't run on import
if __name__ == "__main__":
    results = analyze_fleet("./Dataset/Data/simulated_vibration_small.csv", method="ward", plot=True, distance_threshold=0.6)