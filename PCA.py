#In this project, we will learn about PCA by recreating the Iris dataset

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    mean_vector = np.mean(x, axis = 0)
    #axis = 0 => calculate the mean for each column / axis = 1 => for each row
    #return a vector where each element is the mean for each row
    centered_dataset = x - mean_vector
    return centered_dataset
   

def get_covariance(dataset):
    n = dataset.shape[0]
    d = dataset.shape[1]
    sum = np.zeros((d,d))
  
    for x in dataset:
        sum += np.outer(x,x)
    s = sum / (n-1)
    return s
   

def get_eig(S, m):
    
    e_values, e_vectors = eigh(S)
    
    # Since eigh returns values in ascending order, select the last m values for the top m values
    #[::-1] means to traverse through the list backward
    top_m_values = e_values[-m:][::-1]
    top_m_vectors = e_vectors[:, -m:][:, ::-1]
    
    # Create the diagonal matrix from the top m eigenvalues
    Lambda = np.diag(top_m_values)
    U = top_m_vectors
    
    return Lambda, U
    
    

def get_eig_prop(S, prop):
    e_values, e_vectors = eigh(S)
    total_variance = sum(e_values)
    individual_ration = e_values / total_variance
    
    #return an array of boolean indicating wether the corresponding e_value is greater than p
    selected_index = individual_ration > prop
    selected_e_values = e_values[selected_index]
    selected_e_vectors = e_vectors[:, selected_index]
    
    #since np.linalg.eigh() return e_values in ascending order, traverse through the list backward
    sorted_evalues = selected_e_values[::-1]
    sorted_evectors = selected_e_vectors[:, ::-1]
    
    Lambda = np.diag(sorted_evalues)
    U = sorted_evectors
    
    return Lambda, U
    
   
    

def project_image(image, U):
    #Each image is a 4096 x 1 vector 
    #U is the d x mmatrix that contains the top m eigen vectors 
    #x_pca = UU^Txi
    
    UU_T = U @ np.transpose(U)
    projected = UU_T @ image
    return projected
    
    

def display_image(orig, proj):
    # Please use the format below to ensure grading consistency
    proj_real = np.real(proj)
    orig_reshape = orig.reshape((64, 64))
    proj_reshape = proj_real.reshape((64, 64))
    
    orig_trans = np.transpose(orig_reshape)
    proj_trans = np.transpose(proj_reshape)
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols = 2)
    
    # Display the original image
    im1 = ax1.imshow(orig_trans, aspect='equal')
    ax1.set_title("Original")
    fig.colorbar(im1, ax=ax1)  # Add a colorbar next to the original image
    
    # Display the projected (reconstructed) image
    im2 = ax2.imshow(proj_trans, aspect='equal')
    ax2.set_title("Projection")
    fig.colorbar(im2, ax=ax2)  # Add a colorbar next to the projected image
    
    return fig, ax1, ax2
    
x = load_and_center_dataset('Iris_64x64.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[50], U)
print(projection.dtype)
fig, ax1, ax2 = display_image(x[50], projection)
plt.show()
 
 





  


