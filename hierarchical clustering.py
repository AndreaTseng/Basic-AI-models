#In this project, we implement hierarchical clustering on socioeconomic data from various countries.
#cluster the countries (6 dimensional vector) with hierarchical agglomerative clustering (HAC)
#visualize which countries have similar socioeconomic situations

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    x = open(filepath, 'r')
    list = []
    #skipping the firsr line as it's the column titles
    next(x)
    for line in x:
        dict = {}
        splited_line = line.split(",")
        dict[""] = splited_line[0]
        dict["Country"] = splited_line[1]
        dict["Population"] = splited_line[2]
        dict["Net migration"] = splited_line[3]
        dict["GDP ($ per capita)"] = splited_line[4]
        dict["Literacy (%)"] = splited_line[5]
        dict["Phones (per 1000)"] = splited_line[6]
        dict["Infant mortality (per 1000 births)"] = splited_line[7].strip()
        
        list.append(dict)
    return list

def calc_features(row):
    arr = np.zeros(6)
    i = 0
    j = 0
    for index, (key, value) in enumerate(row.items()):
        if index >= 2:  # Skip the first two key-value pairs
            arr[j] = float(value)
            i += 1
            j += 1
    return arr


def hac(features):
    
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)
    
    def complete_linkage_distance(cluster_a, cluster_b, features):
    
        max_dist = 0
        for point_a in cluster_a:
            for point_b in cluster_b:
                dist = euclidean_distance(features[point_a], features[point_b])
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    
    n = len(features)
    clusters = [[i] for i in range(n)]
    distance_matrix = [[0 if i == j else np.inf for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = distance_matrix[j][i] = complete_linkage_distance(clusters[i], clusters[j], features)
    
    results = []
    clusters_size = len(clusters)
    inactive_clusters = []
    while clusters_size > 1:
        min_dist = np.inf
        closest_pair = (0, 1)
        for i in range(len(distance_matrix)):
            for j in range(i + 1, len(distance_matrix)):
                if distance_matrix[i][j] < min_dist:
                    min_dist = distance_matrix[i][j]
                    closest_pair = (i, j)
   
        cluster_a, cluster_b = closest_pair
        inactive_clusters.append(cluster_a)
        inactive_clusters.append(cluster_b)
        
        new_cluster = clusters[cluster_a] + clusters[cluster_b]
        results.append([cluster_a, cluster_b, min_dist, len(new_cluster)])
        
        clusters.append(new_cluster)
        clusters_size += 1
        
        for dist_list in distance_matrix:
            dist_list.append(0)
        distance_matrix.append([0] * (len(distance_matrix) + 1))
        
        for i in range(len(distance_matrix) - 1):
            dist = complete_linkage_distance(new_cluster, clusters[i], features)
            distance_matrix[-1][i] = distance_matrix[i][-1] = dist

        clusters_size -= 2

        
        current_list = []
        for element in inactive_clusters:
        # Check if the element is a list
            if isinstance(element, list):
                for item in element:
                    current_list.append(item)
            else:
            # If the element is not a list (e.g., an integer), just print it
                current_list.append(element)
        # Mark the merged clusters as inactive
        for i in current_list:
            for j in range(len(distance_matrix)):
                distance_matrix[i][j] = distance_matrix[j][i] = np.inf
        
    
    return np.array(results)

def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 7))  # Adjust the figure size as needed
    dendrogram(Z, labels=names, leaf_rotation=90)  # Rotate the leaf labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated labels
    return fig


def normalize_features(features):
    # Convert the list of numpy arrays into a single numpy array for easier manipulation
    data_matrix = np.array(features)
    
    # Calculate the mean and standard deviation for each column (axis=0 for column-wise operation)
    means = np.mean(data_matrix, axis=0)
    std_devs = np.std(data_matrix, axis=0)
    
    # Normalize the data matrix
    normalized_data_matrix = (data_matrix - means) / std_devs
    
    # Convert the normalized data matrix back into a list of numpy arrays
    normalized_features = [np.array(vector) for vector in normalized_data_matrix]
    
    return normalized_features

data = load_data("countries.csv")
country_names = [row["Country"] for row in data]
features = [calc_features(row) for row in data]
features_normalized = normalize_features(features)
n = 20
Z_raw = hac(features[:n])
Z_normalized = hac(features_normalized[:n])
fig = fig_hac(Z_raw, country_names[:n])
plt.show()

    


