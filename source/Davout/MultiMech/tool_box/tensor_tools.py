# Routine to store tensorial operations of tensor algebra and tensor 
# calculus

from dolfin import *

import ufl_legacy as ufl

from ..tool_box import numerical_tools

# Define the indices for Einstein summation notation

i, j, k, l = ufl.indices(4)

########################################################################
#                                Norms                                 #
########################################################################

# Defines a function to evaluate the L2 norm of a vector

def L2_normVector(vector):

    return numerical_tools.safe_sqrt(ufl.dot(vector, vector))

########################################################################
#                        Skew tensor operations                        #
########################################################################

# Defines a function to construct the skew-symmetric second order tensor
# W from its axial vector

def skew_2OrderTensor(axial_vector):
   
    # Evaluates the permutation third order tensor
    
    perm = ufl.PermutationSymbol(3)

    # Constructs the skew second order tensor W
    
    W = ufl.as_tensor(perm[j,i,k]*axial_vector[k], (i,j))

    return W

# Defines a function to get the tensor product of the axial vector of a
# tensor with the position vector

def tensor_productAxialByPosition(original_tensor, position_vector):
   
    # Evaluates the permutation third order tensor
    
    perm = ufl.PermutationSymbol(3)

    # Gets the axial vector

    axial_vector = ufl.as_vector(0.5*perm[j,i,k]*(original_tensor[j,k]+
    original_tensor[k,j]), (i))

    # Gets the tensor product

    return ufl.as_tensor(axial_vector[i]*position_vector[j], (i,j))

########################################################################
#                               Rotation                               #
########################################################################

# Defines a function to evaluate a rotation second order tensor using 
# the Euler-Rodrigues formula, given the corresponding axial vector, 
# which sets the axis where the rotation is carried about, as well as 
# the angle of rotation in radians from its L2 norm

def rotation_tensorEulerRodrigues(phi, I=Identity(3)):

    # Evaluates the rotation angle

    rotation_angle = L2_normVector(phi)

    # Evaluates the skew tensor

    W = skew_2OrderTensor(phi)

    # Evaluates the Euler-Rodrigues formula

    R_bar = ufl.conditional(gt(rotation_angle, DOLFIN_EPS),((ufl.cos(
    rotation_angle)*I)+((ufl.sin(rotation_angle)/rotation_angle)*W)+(((
    1-ufl.cos(rotation_angle))/(rotation_angle**2))*
    tensor_productVectorVector(phi, phi))), I)

    return R_bar

########################################################################
#                              Projection                              #
########################################################################

# Defines a function to get the projection tensor onto a plane given its
# normal vector

def projection_tensor(n_vector, I=Identity(3)):

    return I-tensor_productVectorVector(n_vector, n_vector)

########################################################################
#                           Tensor products                            #
########################################################################

# Defines the tensor product between two vectors, a and b

def tensor_productVectorVector(a,b):

    C = ufl.as_tensor(a[i]*b[j], (i,j))

    return C