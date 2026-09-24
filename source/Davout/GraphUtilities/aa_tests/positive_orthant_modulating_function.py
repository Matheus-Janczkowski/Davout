# Routine to test the modulating function that sends vectors to the po-
# sitive orthant. Note that this implementation is carried out using 
# numpy, and, for this reason, is a mere test

import numpy as np

# Defines a function to evaluate a vector normal to the identity line in
# n-dimensional real space that belongs to the subspace spanned by the 
# identity line direction vector and a given vector

def get_normal_vector_to_identity_line(u_vector, space_dimension):

    # Gets the inner product of the vector u by the direction of the i-
    # dentity line, d

    u_dot_d = np.sum(u_vector)/np.sqrt(space_dimension)

    # Computes the denominator common to all coefficients of the fol-
    # lowing linear combination

    denominator = np.sqrt(np.dot(u_vector, u_vector)-(u_dot_d*u_dot_d))

    # Checks if the denominator approaches zero

    zero_denominator = np.equal(denominator, 0.0)

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

    coefficient_d = np.where(zero_denominator, 1.0, -((coefficient_u*
    u_dot_d)/np.sqrt(space_dimension)))

    # Returns the combination

    return (coefficient_u*u_vector)+(coefficient_d*np.ones(
    space_dimension))

# Defines a function to get a vector in the boundary of the positive or-
# thant that intercepts the subspace spanned by the given vector u and 
# the identity line. Then, this vector is used to interpolate another 
# vector inside the positive orthant

def normalize_and_rotate_vector_to_positive_orthant(u_vector, 
space_dimension, tolerance=1E-8):

    # Gets the dot product of the given u vector by the positive identi-
    # ty vector

    dimensionality_square_root = np.sqrt(space_dimension)

    u_dot_d = np.sum(u_vector)/dimensionality_square_root

    # Computes the square norm of u

    u_dot_u = np.dot(u_vector, u_vector)

    # Computes beta, that is the argument of the arccosine function

    beta = u_dot_d/np.sqrt(u_dot_u)

    # Evaluates the common denominator of the coefficients to generate
    # a vector perpendicular to the identity line in the subspace span-
    # ned by u and d
    
    denominator = np.sqrt(u_dot_u-(u_dot_d*u_dot_d)) 

    # Checks if the denominator approaches zero
    
    zero_denominator = np.abs(beta+1.0)<=tolerance

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

    coefficient_d = -(coefficient_u*u_dot_d)

    # Builds the vector colinear to the identity line

    d_vector = (1.0/dimensionality_square_root)*np.ones(space_dimension)

    # Gets the orthonormal vector to the identity line

    c_vector = (coefficient_u*u_vector)+(coefficient_d*d_vector)

    # Evaluates the inequation to determine the mu factor

    inequation_numerator = np.abs(c_vector)-c_vector

    inequation_denominator = (2.0*(d_vector-c_vector))

    mu_inequation = np.divide(inequation_numerator, 
    inequation_denominator, out=np.zeros_like(inequation_numerator), 
    where=inequation_denominator!=0)

    # Gets the maximum mu factor and the corresponding boundary vector

    mu = np.max(mu_inequation)

    # Gets the relative position of the vector u with respect to the i-
    # dentity line. The relative position must be 1 when it lies on the
    # identity line and 0 when it lies on the negative identity line. For
    # this purpose, we use beta, since it is contained within the inter-
    # val [-1,1]

    ratio = 0.5*(beta+1.0)

    # Gets the mu corresponding to the final vector inside the positive
    # orthant

    final_mu = (mu*(1-ratio))+ratio

    denominator = np.sqrt((final_mu**2)+((1.0-final_mu)**2))

    # Constructs the final vector as a linear interpolation of the iden-
    # tity line and the orthonormal vector c

    return ((final_mu/denominator)*d_vector)+(((1.0-final_mu)/
    denominator)*c_vector)

# Defines a function to get any vector u in n-dimensional real space,
# make it unitary, and then rotate it to the positive orthant inside a
# 2-dimensional subspace spanned by the original vector u and the posi-
# tive and unit identity vector

def normalize_and_rotate_vector_to_positive_orthant_cone(u_vector,
space_dimension, tolerance=1E-10):

    # Gets the dot product of the given u vector by the positive identi-
    # ty vector

    dimensionality_square_root = np.sqrt(space_dimension)

    u_dot_d = np.sum(u_vector)/dimensionality_square_root

    # Computes the square norm of u

    u_dot_u = np.dot(u_vector, u_vector)

    # Computes beta, that is the argument of the arccosine function

    beta = u_dot_d/np.sqrt(u_dot_u)

    # Computes the quarter cosine and the quarter sine

    inner_square_root = np.sqrt(0.5*(1.0+beta))

    quarter_cosine = np.sqrt(0.5*(1+inner_square_root))

    quarter_sine = np.sqrt(0.5*(1-inner_square_root))

    # Evaluates the common denominator of the coefficients to generate
    # a vector perpendicular to the identity line in the subspace span-
    # ned by u and d
    
    denominator = np.sqrt(u_dot_u-(u_dot_d*u_dot_d)) 

    # Checks if the denominator approaches zero
    
    zero_denominator = np.abs(beta+1.0)<=tolerance

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

    coefficient_d = -((coefficient_u*u_dot_d)/dimensionality_square_root)

    # Assembles the resulting rotated vector

    return (((quarter_cosine+(coefficient_d*quarter_sine))/
    dimensionality_square_root)*np.ones(space_dimension)+(coefficient_u*
    quarter_sine*u_vector))

# Defines a function to plot 

# Testing block

if __name__=="__main__":

    # Gets the vector u

    u = np.asarray([-0.5, -0.5, 1.0])

    # Gets the normal vector to the identity line that is spanned by u
    # and the vector u

    c = get_normal_vector_to_identity_line(u, u.shape[0])

    print("\nThe normal vector given by "+str(u)+" is: "+str(c))

    # Evaluates the corresponding vector that shall be rotated to the 
    # positive orthant

    positive_u = normalize_and_rotate_vector_to_positive_orthant(u,
    u.shape[0])

    print("\nThe rotated vector to the positive orthant given by "+str(u
    )+" is: "+str(positive_u)+"\nwhose norm is "+str(np.linalg.norm(
    positive_u)))