import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#########################################
# Function to get the covariance matrix #
#########################################

def get_covariance_matrix(data):
    return np.dot(data.T, data)/data.shape[0]

######################################################################
# Function to get the components, given a certain fidelity threshold #
######################################################################

def get_n_components(sorted_values, fidelity_threshold):
    i=1
    eigen_sum = np.sum(sorted_values)
    partial_sum = sorted_values[0]
    while (partial_sum/eigen_sum) < fidelity_threshold:
        partial_sum += sorted_values[i]
        i+=1
    return i

#######################################################################
# Function to perform the PCA. The number of components can be either #
# set by the user or it can be found automatically based on the       #
# fidelity threshold.                                                 #
#######################################################################

def pca(data, n_components=None, fidelity_threshold=0.95, plot_spectrum = False):

    covariance_matrix = get_covariance_matrix(data)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # if the number of principal components is not provided by the user, get it automatically
    if n_components is None:
        n_components=get_n_components(sorted_eigenvalues, fidelity_threshold)

    if plot_spectrum:
        # plotting the eigenvalues
        x=np.arange(0,len(sorted_eigenvalues))
        plt.figure()
        plt.semilogy()
        plt.title('Spectrum of the eigenvalues in logscale')
        plt.plot(x[:n_components], sorted_eigenvalues[:n_components], 'o', color = "orange", label="Selected eigenvalues")
        plt.plot(x[n_components:], sorted_eigenvalues[n_components:], 'o')
        plt.legend()
        plt.show()

    # get principal components
    pca_components = sorted_eigenvectors[:, :n_components]
    transformed_data = np.dot(data, pca_components)

    return transformed_data, pca_components

###################################################################
# Set of functions to calculate the normalized mutual information #
###################################################################

def shan_entropy(X, n_classes):
    P_X = np.histogram(X, n_classes)[0]
    P_X = P_X/np.sum(P_X)
    H_X = -np.sum(P_X * np.log(P_X))
    return H_X

def normalized_mutual_information(X, Y, n_classes):
    P_xy = np.histogram2d(X, Y, n_classes)[0]
    P_xy=P_xy/np.sum(P_xy)
    P_x = np.histogram(X, n_classes)[0]
    P_x = P_x/np.sum(P_x)
    P_y = np.histogram(Y, n_classes)[0]
    P_y=P_y/np.sum(P_y)
    MI = 0.0
    for i in range(P_xy.shape[0]):
        for j in range(P_xy.shape[1]):
            if P_xy[i,j] > 0.0:
                MI += P_xy[i,j] * np.log(P_xy[i,j] / (P_x[i] * P_y[j]))
    # H_X = shan_entropy(X, n_classes)
    # H_Y = shan_entropy(Y, n_classes)
    # max_MI = min(H_X, H_Y)
    # if max_MI > 0.0:
    #    return MI / max_MI
    #else:
    #    return 0.0
    return MI
    
#######################################
# Functions to perform the kernel-PCA #
#######################################

# Compute the distance matrix
def get_distance_matrix(X):
    A = X[:, np.newaxis, :]
    B = X[np.newaxis, :, :]
    return np.sqrt(np.sum(np.square(A-B), axis=-1))

def gaussian_kernel(distance_matrix, sigma):
    return np.exp(-(distance_matrix**2)/(2*(sigma**2)))

def get_double_centered_Gram(K):
    N = K.shape[0]
    C = np.sum(K)/N**2
    B = np.add(np.zeros_like(K), np.sum(K, axis=1))
    B = np.add(B, B.T)
    B /= N
    return K - B + C


def k_pca(data, n_components=2, sigma=1.0):

    K = gaussian_kernel(get_distance_matrix(data), sigma=sigma)
    G = get_double_centered_Gram(K)
    
    eigenvalues, eigenvectors = sp.linalg.eigh(G, subset_by_index=(len(G)-n_components, len(G)-1))
    sorted_eigenvalues = eigenvalues[::-1]
    sorted_eigenvectors = eigenvectors[:,::-1]

    Y = np.multiply(sorted_eigenvectors, np.sqrt(sorted_eigenvalues))
    
    return Y


###############################################
# Functions to implement the DBSCAN algorithm #
###############################################

def find_neighbors(X, center_id, eps):
    neighbors = []
    center_point = X[center_id]
    for i, point in enumerate(X):
        if np.linalg.norm(point-center_point) <= eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(X, labels, center_id, neighbors, cluster_label, eps, min_pts):
    labels[center_id] = cluster_label
    i = 0
    while i < len(neighbors):
        neighbor_id = neighbors[i]
        
        if labels[neighbor_id] == -1:  #noise label becomes a border point
            labels[neighbor_id] = cluster_label

        elif labels[neighbor_id] == 0:
            labels[neighbor_id] = cluster_label
            new_neighbors = find_neighbors(X, neighbor_id, eps)
            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)

        i += 1


def dbscan(X, eps, min_pts):
    n_samples = X.shape[0]
    labels = np.zeros(n_samples, dtype=int)  #0 for unvisited points, -1 noise points, positive integers for clusters
    cluster_label = 0
    for i in range(n_samples):
        if labels[i] != 0:  #already visited 
            continue
        neighbors = find_neighbors(X, i, eps)

        if len(neighbors) < min_pts:
            labels[i] = -1  #assign noise label
        else:
            cluster_label += 1
            expand_cluster(X, labels, i, neighbors, cluster_label, eps, min_pts)
    return labels

#####################
# K-means algorithm #
#####################

def kmeans(X, k, max_iterations=10000):
    n_samples, _ = X.shape

    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iterations):
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_samples = X[labels == i]
            if len(cluster_samples) > 0:
                new_centroids[i] = cluster_samples.mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids



#########################
# Clustering validation #
#########################

def silhouette_score(X, labels):
    '''
    Function to calculate the average of the silhoutte coefficients
    '''
    n_samples = X.shape[0]
    silhouette_coefficients = np.zeros(n_samples)
    
    for i in range(n_samples):
        mask = (labels==labels[i])
        a_i = np.mean(np.linalg.norm(X[mask] - X[i], axis=1)) #cohesion, average distance from vectors belonging to the same clusters
        
        unique_labels = np.unique(labels)
        b_i = np.inf #separation

        for label in unique_labels:
            if label == labels[i]:
                continue
            
            mask = (labels==label)
            temp = np.mean(np.linalg.norm(X[mask] - X[i], axis=1)) #average distance from vectors belonging to other clusters
            if temp < b_i:
                b_i = temp
        silhouette_coefficients[i] = (b_i - a_i) / max(a_i, b_i)
        
    return np.mean(silhouette_coefficients)


def kmeans_objective(X, labels, centroids):
    
    n_samples, _ = X.shape
    objective = 0.0

    for i in range(n_samples):
        centroid = centroids[labels[i]]
        distance = np.linalg.norm(X[i] - centroid)
        objective += distance ** 2

    return objective


#######################################
# Functions for Markov State Modeling #
#######################################


def estimate_transition_prob_matrix(labels, tau):

    n_states = np.max(labels) + 1
    P = np.zeros(shape=(n_states, n_states))

    #we count each transition from state i to j for a time lag of tau
    ending_states = labels[tau:]
    for (i, j) in zip(labels, ending_states):
        P[i, j] += 1
    
    #we now make it a probability
    row_sums = P.sum(axis=1)
    P = P/row_sums[:,np.newaxis]

    return P