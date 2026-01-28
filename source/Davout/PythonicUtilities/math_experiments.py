import numpy as np

# Defines a function to generate a set of orthonormal vectors from a set
# of vectors

def gram_schmidt_orthogonalization(list_of_vectors):

    # Iterates through the vectors

    for i in range(len(list_of_vectors)):

        # Iterates through the vectors that came befor

        for j in range(i):

            # Subtracts this vector from the current one

            list_of_vectors[i] -= np.dot(list_of_vectors[j], 
            list_of_vectors[i])*list_of_vectors[j]

        # Normalizes the current vector

        list_of_vectors[i] = list_of_vectors[i]/np.linalg.norm(
        list_of_vectors[i])

    return list_of_vectors

# Defines a function to generate a orthonormal matrix from a upper tria-
# gonal matrix with another subdiagonal under the diagonal

def generate_orthonormal_from_quasi_triangular(dimension):

    # Initializes a list of vectors

    list_of_columns = []

    # Iterates through the columns

    for i in range(dimension):

        # Generates a zero vector

        v = np.zeros(dimension)

        # Iterates through the nonzero terms

        for j in range(min(i+2, dimension)):

            # Adds the nonzero term

            v[j] = np.random.rand()

        # Appends this vector to the list

        list_of_columns.append(v)

    # Generates the orthonormal basis from these columns

    print(list_of_columns)

    list_of_columns = gram_schmidt_orthogonalization(list_of_columns)

    print(list_of_columns)

# Defines a function to get the SVD decomposition of a random matrix 
# with non-negative entries

def SVD_decomposition(n_rows, n_columns, n_tests, amplitude_singular):

    A = np.random.rand(n_rows, n_columns)

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print("U.shape="+str(U.shape))

    print("S.shape="+str(S.shape))

    print("Vt.shape="+str(Vt.shape))

    print("\nA="+str(A))

    print("\nU="+str(U))

    print("\nS="+str(S))

    print("\nV.T="+str(Vt))

    rank = min(n_rows, n_columns)

    for i in range(n_tests):

        for j in range(rank):

            S[j] = amplitude_singular*np.random.rand()

        print("\nS="+str(S))

        print("A="+str(np.dot(U, np.dot(np.diag(S), Vt))))

########################################################################
#                               Testing                                #
########################################################################

def test_gram_schmidt():

    d = 2

    n_samples = 3

    list_of_vectors = [np.random.randn(d) for i in range(n_samples)] 

    list_of_vectors = gram_schmidt_orthogonalization(list_of_vectors)

    print(list_of_vectors)

    # Tests the orthogonality

    error = 0.0

    for i in range(n_samples):

        for j in range(n_samples):

            if i==j:

                error += (np.dot(list_of_vectors[i], list_of_vectors[j])
                -1.0)**2

            else:

                error += (np.dot(list_of_vectors[i], list_of_vectors[j])
                )**2

    print("The RMS error is "+str(np.sqrt(error/(n_samples*n_samples))))

def test_orthonormal_basis():

    d = 4

    generate_orthonormal_from_quasi_triangular(d)

def test_orthonormal_basis():

    n_rows = 5

    n_columns = 4

    n_tests = 5

    amplitude_singular = 5.0

    SVD_decomposition(n_rows, n_columns, n_tests, amplitude_singular)

if __name__=="__main__":

    test_gram_schmidt()
    
    test_orthonormal_basis()