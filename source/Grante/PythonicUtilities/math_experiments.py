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
            list_of_vectors[i])*list_of_vectors[i]

        # Normalizes the current vector

        list_of_vectors[i] = list_of_vectors[i]/np.linalg.norm(
        list_of_vectors[i])

    return list_of_vectors

########################################################################
#                               Testing                                #
########################################################################

def test_gram_schmidt():

    d = 4

    list_of_vectors = [np.random.randn(d) for i in range(d)] 

    list_of_vectors = gram_schmidt_orthogonalization(list_of_vectors)

    # Tests the orthogonality

    error = 0.0

    for i in range(d):

        for j in range(d):

            if i==j:

                error += (np.dot(list_of_vectors[i], list_of_vectors[j])
                -1.0)**2

            else:

                error += (np.dot(list_of_vectors[i], list_of_vectors[j])
                )**2

    print("The RMS error is "+str(np.sqrt(error/())))

if __name__=="__main__":

    test_gram_schmidt()