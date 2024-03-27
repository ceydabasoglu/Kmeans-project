import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("midtermProject-part2-data.csv")

# Function to normalize the data
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Function to perform k-means clustering
def perform_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans.cluster_centers_

# Function to calculate Within Cluster Sum of Squares (WCSS)
def calculate_wcss(data, clusters, centers):
    wcss = 0
    data_df = pd.DataFrame(data)  # Convert NumPy array to Pandas DataFrame
    for i in range(len(data_df)):
        wcss += np.linalg.norm(data_df.iloc[i] - centers[clusters[i]])**2
    return wcss


# Function to calculate Between Cluster Sum of Squares (BCSS)
def calculate_bcss(data, clusters, centers):
    bcss = 0
    overall_center = np.mean(data, axis=0)  # Calculate mean without using .values
    for i in range(len(centers)):
        bcss += len(data[clusters == i]) * np.linalg.norm(centers[i] - overall_center)**2
    return bcss

# Function to calculate Dunn Index
def calculate_dunn_index(data, clusters, centers):
    min_intercluster_distance = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            intercluster_distance = np.linalg.norm(centers[i] - centers[j])
            if intercluster_distance < min_intercluster_distance:
                min_intercluster_distance = intercluster_distance

    max_intracluster_diameter = -np.inf
    for i in range(len(centers)):
        cluster_points = data[clusters == i]
        intracluster_diameter = pairwise_distances(cluster_points).max()
        if intracluster_diameter > max_intracluster_diameter:
            max_intracluster_diameter = intracluster_diameter

    dunn_index = min_intercluster_distance / max_intracluster_diameter
    return dunn_index

# Function to save results to result.txt
def save_results(clusters, wcss, bcss, dunn_index):
    with open("result.txt", "w") as file:
        for i, cluster in enumerate(clusters):
            file.write(f"Record {i + 1} : Cluster {cluster + 1}\n")

        for i in range(len(set(clusters))):
            file.write(f"Cluster {i + 1} : {np.sum(clusters == i)} records\n")

        file.write(f"\nWCSS : {wcss:.2f}\n")
        file.write(f"BCSS : {bcss:.2f}\n")
        file.write(f"Dunn Index : {dunn_index:.2f}\n")

# Function for cluster visualization
def visualize_clusters(data, clusters, x_variable, y_variable):
    if x_variable is None or y_variable is None:
        print("Invalid input for variable names. Visualization aborted.")
        return

    plt.scatter(data[x_variable], data[y_variable], c=clusters, cmap='viridis')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.title("Cluster Visualization")
    plt.show()

# Main program
def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    # Get user input for k
    k = int(input("Enter the value of k for k-means clustering: "))
    
    
    # Normalize the data
    normalized_data = normalize_data(df.iloc[:, 1:])

    # Perform clustering
    clusters, centers = perform_clustering(normalized_data, k)

    # Calculate WCSS, BCSS, and Dunn Index
    wcss = calculate_wcss(normalized_data, clusters, centers)
    bcss = calculate_bcss(normalized_data, clusters, centers)
    dunn_index = calculate_dunn_index(normalized_data, clusters, centers)

    # Save results to result.txt
    save_results(clusters, wcss, bcss, dunn_index)

    # Cluster visualization
    x_variable = input("Enter the variable for the x-axis: ").strip()
    y_variable = input("Enter the variable for the y-axis: ").strip()
    visualize_clusters(df, clusters, x_variable, y_variable)

if __name__ == "__main__":
    main()
