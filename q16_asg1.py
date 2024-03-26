import numpy as np

def jacobi(coefficients, constants, initial_guess,true_sol,  tolerance=0.01, max_iterations=1000):
    n = len(constants)
    x = initial_guess.copy()  #copies initial guess into a different array x
    x_new = np.zeros_like(x)  #generates an array of zeros that matches the shape and data type of x
    iteration = 0
    while iteration < max_iterations:
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += coefficients[i][j] * x[j]  #a part of the formula for jacobi method
            x_new[i] = (constants[i] - sigma) / coefficients[i][i]  #next iterative solution 
        if np.linalg.norm(true_sol - x_new) < tolerance:  #convergence condition
            break
        x = x_new.copy()
        iteration += 1
    return x_new, iteration

def gauss_seidel(coefficients, constants, initial_guess, true_sol, tolerance=0.01, max_iterations=1000):
    n = len(constants)
    x = initial_guess.copy()
    iteration = 0
    while iteration < max_iterations:
        x_new = x.copy()
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += coefficients[i][j] * x_new[j]
            x_new[i] = (constants[i] - sigma) / coefficients[i][i] #iterative solution
        if np.linalg.norm(true_sol - x_new) < tolerance: #convergence condition
            break
        x = x_new.copy()
        iteration += 1
    return x_new, iteration

def relaxation(coefficients, constants, initial_guess, true_sol,  w=1.25, tolerance=0.01, max_iterations=1000):
    n = len(constants)
    x = initial_guess.copy()
    iteration = 0
    res = np.zeros(n) # residual vector
    while iteration < max_iterations:
        x_new = x.copy()
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += coefficients[i][j] * x_new[j]
            res[i]= constants[i] - sigma - (coefficients[i][i] * x[i])        
            x_new[i] = x[i] + w * (res[i] / coefficients[i][i]) #w=relaxation constant
        if np.linalg.norm(true_sol - x_new) < tolerance: #convergence condition
            break
        x = x_new.copy()
        iteration += 1
    return x_new, iteration


def conjugate_gradient(A, b, x0, tolerance=0.01, max_iterations=1000):
    n = len(b)
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    iteration = 0
    while iteration < max_iterations:
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tolerance:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        iteration += 1
    return x, iteration

# Coefficients matrix
coefficients = np.array([[0.2, 0.1, 1, 1, 0],
                         [0.1, 4, -1, 1, -1],
                         [1, -1, 60, 0, -2],
                         [1, 1, 0, 8, 4],
                         [0, -1, -2, 4, 700]])

# Constants vector
constants = np.array([1, 2, 3, 4, 5])

# Initial guess
initial_guess = np.zeros(len(constants))

#given true solution
true_solution = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])

# Jacobi method
solution_jacobi, iterations_jacobi = jacobi(coefficients, constants, initial_guess, true_solution)
print("Jacobi Method:")
print("Solution:", solution_jacobi)
print("Number of iterations:", iterations_jacobi)

# Gauss-Seidel method
solution_gauss_seidel, iterations_gauss_seidel = gauss_seidel(coefficients, constants, initial_guess, true_solution)
print("\nGauss-Seidel Method:")
print("Solution:", solution_gauss_seidel)
print("Number of iterations:", iterations_gauss_seidel)

# Relaxation method
solution_relaxation, iterations_relaxation = relaxation(coefficients, constants, initial_guess, true_solution, w=1.25)
print("\nRelaxation Method (w = 1.25):")
print("Solution:", solution_relaxation)
print("Number of iterations:", iterations_relaxation)

# Conjugate Gradient method
solution_cg, iterations_cg = conjugate_gradient(coefficients, constants, initial_guess)
print("\nConjugate Gradient Method:")
print("Solution:", solution_cg)
print("Number of iterations:", iterations_cg)
