# Routine to demonstrate the differences between the Green-Lagrange and
# infinitesimal strain tensors

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import numpy as np

# Defines a function to evaluate the rotation tensor as a function of 
# time

def rotation_tensor(t):

    theta = 0.5*np.pi*t

    c = np.cos(theta)

    s = np.sin(theta)

    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

# Defines a function to evaluate the stretch tensor 

def stretch_tensor(t):

    T = min(t, 1.0)

    return np.array([[T, 0.0, 0.0], [0.0, T*T, 0.0], [0.0, 0.0, 1.0]])

# Defines the displacement as a linear mapping

def displacement_field(t, material_position_vector):

    # Evaluates the rotation and stretch tensors

    R = rotation_tensor(t)

    U = stretch_tensor(t)

    # Evaluates the displacement field

    return np.einsum('ik,k->i', np.einsum('ij,jk->ik', R, U)-np.eye(3), 
    material_position_vector)

# Defines a function to evaluate the deformation gradient

def deformation_gradient(t):

    # Evaluates the rotation and stretch tensors

    R = rotation_tensor(t)

    U = stretch_tensor(t)

    return np.matmul(R, U)+np.eye(3)

# Defines a function to compute the Green-Lagrange strain tensor

def green_lagrange_strain(t):

    # Evaluates the deformation gradient

    F = deformation_gradient(t)

    # Evaluates the Green-Lagrange strain tensor

    return 0.5*(np.einsum('ji,jk->ik', F, F)-np.eye(3))

# Defines a function to compute the infinitesimal strain tensor

def infinitesimal_strain(t):

    # Evaluates the deformation gradient

    F = deformation_gradient(t)

    return 0.5*(F.T+F)-np.eye(3)

# Defines a function to compute the Frobenius norm of the difference of
# the two tensors

def difference_between_strain_tensors(t):

    # Computes the Green-Lagrange and infinitesimal strain tensors

    E = green_lagrange_strain(t)

    epsilon = infinitesimal_strain(t)

    # Returns the Frobenius norm

    delta = E-epsilon

    return np.linalg.norm(delta, ord="fro")

# Test block

if __name__=="__main__":

    # Gets a set of time points

    t_final = 1.0

    t_initial = 0.0

    t_range = np.linspace(t_initial, t_final, 50)

    # Computes the difference between the two tensor as a function of time

    differences = [difference_between_strain_tensors(t) for t in t_range]

    """plt.figure(figsize=(8, 5))

    plt.plot(t_range, differences, color="crimson", linewidth=2)

    plt.xlabel("Time $t$ / Deformation Magnitude", fontsize=11)

    plt.ylabel(r"Frobenius Norm $\|\mathbf{E} - \mathbf{\epsilon}\|_F$",
    fontsize=11)

    plt.title("Divergence between Green-Lagrange and Infinitesimal Str"+
    "ain", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.show()"""

    x = [500.0, 850.0, 1200.0]

    y = [3.424, 5.711, 8.086]

    x = np.array([[x_i] for x_i in x])

    y = np.array(y)

    model = LinearRegression()

    model.fit(x, y)
    
    print("Slope: "+str(model.coef_[0]))