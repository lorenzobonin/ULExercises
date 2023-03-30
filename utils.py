import numpy as np
import matplotlib.pyplot as plt

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
    
    # plotting the eigenvalues
    x=np.arange(0,len(sorted_eigenvalues))

    if plot_spectrum:
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
    H_X = shan_entropy(X, n_classes)
    H_Y = shan_entropy(Y, n_classes)
    max_MI = min(H_X, H_Y)
    if max_MI > 0.0:
        return MI / max_MI
    else:
        return 0.0
    
######################################
# Function to perform the kernel-PCA #
######################################
def gaussian_kernel_matrix(X, sigma=1.0):
    n_samples = X.shape[0]
    X_norms = np.linalg.norm(X, axis=0)
    X_norms = np.tile(X_norms, (n_samples, 1))
    X = X / X_norms
    K = np.exp(-np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1) / (2 * (sigma ** 2)))
    np.fill_diagonal(K, 1)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    return K


def k_pca(data, n_components=None, fidelity_threshold=0.95, sigma=1.0, plot_spectrum = False):

    
    data = gaussian_kernel_matrix(data, sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(data)

    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # if the number of principal components is not provided by the user, get it automatically
    if n_components is None:
        n_components=get_n_components(sorted_eigenvalues, fidelity_threshold)
    
    # plotting the eigenvalues
    if plot_spectrum:
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