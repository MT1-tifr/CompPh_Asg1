import numpy as np

# Define the matrix A
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

# Function to find the dominant eigenvalue and eigenvector using Power Method
def power_method(matrix, tolerance=0.01, max_iterations=1000):
    n = matrix.shape[0] #assigning the no. of rows of the matrix to n
    x = np.ones(n)  # Initial guess for the eigenvector
    eigenvalue_old = 0
    
    for _ in range(max_iterations):
        x_new = np.dot(matrix, x)
        eigenvalue_new = np.dot(x, x_new) / np.dot(x, x)
        x_new /= np.linalg.norm(x_new)
        
        if np.abs(eigenvalue_new - eigenvalue_old) < tolerance:
            break
        
        eigenvalue_old = eigenvalue_new
        x = x_new
    
    return eigenvalue_new, x_new

# Apply Power Method to find dominant eigenvalue and eigenvector
eigenvalue, eigenvector = power_method(A)

# Print results
print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)
