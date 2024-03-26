import numpy as np

def qr_eigenval(matrix, max_iter=100, tol=1e-6): 
    n = matrix.shape[0] 
    V = np.eye(n) #creating an identity matrix of order n
    A=matrix.copy()
    
    for _ in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)
        V = np.dot(V, Q)
        if np.abs(A.diagonal(-1)).max() < tol: #diagonal(-1) gives the elements just below the diagonal of the matrix
            break

    eigenvalues = A.diagonal()
    return eigenvalues, V

# Given matrix
A=np.array([[5,-2],[-2,8]])

# Calculate eigenvalues using QR decomposition
eigenvalues, eigenvectors = qr_eigenval(A)

print("Eigenvalues using QR decomposition:")
print(eigenvalues)


# Eigenvalues calculation using numpy.linalg.eigh
eigenvalues_eigh = np.linalg.eigh(A)[0]

print("\nEigenvalues calculated using numpy.linalg.eigh:")
print(eigenvalues_eigh)
