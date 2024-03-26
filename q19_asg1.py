import numpy as np
import time

# Defining the matrices
matrices = [
    np.array([[2, 1],
              [1, 0]]),
    
    np.array([[2, 1],
              [1, 0],
              [0, 1]]),
    
    np.array([[2, 1, -1, 1],
              [-1, 1, 1, 2],
              [1, 1, 2, -1]]),
    
    np.array([[1, 1, 0],
              [-1, 0, 1],
              [0, 1, -1],
              [1, 1, -1]]),
    
    np.array([[0, 1, 1],
              [0, 1, 0],
              [1, 1, 0],
              [0, 1, 0],
              [1, 0, 1]])
]

# Performing SVD for each matrix and reporting the time taken
for i, matrix in enumerate(matrices):
    print("\nOriginal Matrix:")
    print(matrix)
    print(f"\nSVD for Matrix {i + 1}:")
    start_time = time.time()
    U, Sigma, Vt = np.linalg.svd(matrix)  #sigma will only give the diagonal values
    print('U=',U) #m by m matrix
   # Create zero matrix with original matrix dimensions
    S_matrix = np.zeros_like(matrix, dtype=float)
    S_matrix[:min(matrix.shape), :min(matrix.shape)] = np.diag(Sigma) #creates m by n matrix with only diagonal entries
    print()
    print('S=',S_matrix)
    print()
    print('Vt=',Vt) #n by n matrix
    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")
    Recon_matrix=np.dot(U,np.dot(S_matrix,Vt))
    print("Reconstructed matrix=",Recon_matrix)
    

