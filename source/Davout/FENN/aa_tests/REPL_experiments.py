import tensorflow as tf

import numpy as np

# Defines a function to test the gather function

def test_gather_vector():

    # Constructs a vector 

    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Constructs a vector of indices

    indices = tf.constant([0, 2])

    print("\nTests the gather function with the vector b="+str(b.numpy()
    )+" and the indices="+str(indices.numpy())+":\n"+str(tf.gather(b, 
    indices).numpy()))

# Defines a function to test the gather function for accessing a matrix

def test_gather_tensor():

    # Constructs a tensor of shape [n_elements, n_quadrature_points, 3, 
    # 3]

    n_elements = 3

    n_quadrature_points = 2

    F = tf.random.uniform([n_elements, n_quadrature_points, 3, 3])

    indices = tf.constant([0, 2])

    print("\nTests the gather function for the tensor:\n"+str(F.numpy()
    )+"\n\nindices:\n"+str(indices.numpy())+"\n\nresult is:\n"+str(
    tf.gather(F, indices).numpy())+"\n")

# Defines a function to test indexing the last index

def test_last_index():

    n_quadrature_points = 4

    points = tf.random.uniform([n_quadrature_points, 3])

    print("\nTests indexing the last index and putting '...' in the fi"+
    "rst index slot. The original tensor is:\n"+str(points)+"\n\nThe i"+
    "ndexed result is:\n"+str(points[...,2].numpy())+"\n")

def get_tetra_coeffs():

    row = lambda r, s, t: np.array([r**2, s**2, t**2, r*s, r*t, s*t, r, 
    s, t, 1.0])

    terms = ["r^2", "s^2", "t^2", "r*s", "r*t", "s*t", "r", "s", "t", ""]

    nodes = [[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0.5,0.5,0],[0,0.5,0.5],[
    0,0,0.5],[0.5,0,0],[0.5,0,0.5],[0,0.5,0]]

    A = np.zeros((10,10))

    for i in range(10):

        A[i,:] = row(*nodes[i])

    for i in range(10):

        b = np.zeros(10)

        b[i] = 1.0

        coeffs = np.linalg.solve(A, b)

        shape_function = ""

        for j in range(10):

            if abs(coeffs[j])>1E-5:

                shape_function += str(coeffs[j])+"*"+str(terms[j])+" + "

        print("The "+str(i+1)+" shape function is:\n"+shape_function[0:-2
        ]+"\n\n")

    row = lambda r, s, t: np.array([(2*r*r)-r, (2*s*s)-s, (2*t*t)-t, (2*
    ((r*r)+(s*s)+(t*t)))-(3*(r+s+t))+(4*((r*s)+(r*t)+(s*t)))+1.0, 4*r*s, 
    4*s*t, -(4*t*t)-(4*r*t)-(4*s*t)+(4*t), -(4*r*r)-(4*r*s)-(4*r*t)+(4*r
    ), 4*r*t, -(4*s*s)-(4*r*s)-(4*s*t)+(4*s)])

    for i in range(10):

        A[i,:] = row(*nodes[i])

    print("The verification matrix is:\n"+str(A)+"\n\n")

def get_quadrature_points():

    points = [-1/np.sqrt(3), 1/np.sqrt(3)]

    weights = [1.0, 1.0]

    n_points = 2

    tetra_points = []

    tetra_weights = []

    string = ""

    for i in range(n_points):

        for j in range(n_points):

            for k in range(n_points):

                point_r = (0.5*(1.0+points[i]))

                point_s = (0.25*(1.0-points[i])*(1.0+points[j]))

                point_t = (0.125*(1.0+points[i])*(1.0-points[j])*(1.0+
                points[k]))

                weight = ((((1.0-points[j])*((1.0-points[i])**2))/64.0)*
                weights[i]*weights[j]*weights[k])

                tetra_weights.append(weight)

                tetra_points.append([point_r, point_s, point_t])

                string += "\nw="+str(tetra_weights[-1])+"; point="+str(
                tetra_points[-1])

    print("The Gauss points for a tetrahedron with "+str(n_points)+" p"+
    "oints in each direction are:"+string)

    def integral_test(integrand, points, weights):

        result = 0.0

        for weight, point in zip(weights, points):

            result += (weight*integrand(*point))

        return result 
    
    print("\nThe volume of the tetrahedron given this rule is "+str(
    integral_test(lambda x, y, z: 1.0, tetra_points, tetra_weights))+
    "; while 1/6="+str(1/6))

    alpha = 2

    beta = 1

    gamma = 3

    print("\nThe integration of (r**"+str(alpha)+")*(s**"+str(beta)+")"+
    "*(t**"+str(gamma)+")="+str(integral_test(lambda r, s, t: (r**alpha
    )*(s**beta)*(t**gamma), tetra_points, tetra_weights)))

# Defines a function to test what newaxis does

def test_shape_functions_in_natural_coordinates():

    # Defines Gauss points in a 2D space

    gauss_points = tf.constant([[1./3, 1./3], [0.2, 0.2], [0.6, 0.2], [
    0.2, 0.6]])

    # Gets the individual values

    r = gauss_points[...,0]

    s = gauss_points[...,1]

    s0 = (1. - r - s)[..., tf.newaxis]

    print("\nr="+str(r)+", s="+str(s)+",\ns0 = (1. - r - s)[..., tf.ne"+
    "waxis]:\n"+str(s0)+"\n")

    # Gets more values

    s1 = r[..., tf.newaxis]

    s2 = s[..., tf.newaxis]

    print("s1 = r[..., tf.newaxis]:\n"+str(s1)+"\n\n")

    print("s2 = s[..., tf.newaxis]:\n"+str(s2)+"\n\n")

    # Concatenates s0, s1, and s2

    shape_fn = tf.concat((s0, s1, s2), axis=-1)

    print("shape_fn = tf.concat((s0, s1, s2), axis=-1):\n"+str(shape_fn
    )+"\n")

    dsdr = tf.concat([-tf.ones_like(s0), tf.ones_like(s1), tf.zeros_like(
    s2)], axis=-1)

    dsds = tf.concat([-tf.ones_like(s0), tf.zeros_like(s1), tf.ones_like(
    s2)], axis=-1)

    print("dsdr:\n"+str(dsdr)+"\n\ndsds:\n"+str(dsds)+"\n")

    shape_fn_grad = tf.concat((dsdr[tf.newaxis, ...], dsds[tf.newaxis, 
    ...]), axis=0)

    print("shape_fn_grad:\n"+str(shape_fn_grad)+"\n")

    # Defines a tensor of nodes coordinates. The first index is for the
    # elements; the second index is for the local number of the node; the
    # third is for the value of the coordinate itself. There follows an
    # example for a mesh of two triangular elements in a square domain

    nodes = tf.constant([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[0.0, 
    0.0], [1.0, 1.0], [0.0, 1.0]]])

    # Gets the x and y coordinates of the nodes. Adds the new axis in the
    # middle to compatibilize it with the dimension of quadrature points

    x = nodes[..., 0][..., tf.newaxis, :]

    y = nodes[..., 1][..., tf.newaxis, :]

    print("The shape of the tensor of nodes coordinates is:\n"+str(
    nodes.shape)+"\n\nThe x coordinates of the nodes are:\n"+str(x)+
    "\n\nThe y coordinates of the nodes are:\n"+str(y)+"\n")

    # Computes the jacobian of the transformation from original coordi-
    # nates to the natural coordinates

    print("shape_fn_grad[0]:\n"+str(shape_fn_grad[0])+"\n")

    print("shape_fn_grad[0]*x:\n"+str(shape_fn_grad[0]*x)+"\n")

    j11 = tf.reduce_sum(shape_fn_grad[0]*x, axis=-1)

    j12 = tf.reduce_sum(shape_fn_grad[0]*y, axis=-1)

    j21 = tf.reduce_sum(shape_fn_grad[1]*x, axis=-1)

    j22 = tf.reduce_sum(shape_fn_grad[1]*y, axis=-1)

    jacobian_det = j11*j22 - j12*j21

    print("The determinant of the jacobian is:\n"+str(jacobian_det)+
    "\n")

    # Computes the derivative of the shape functions with respect to the
    # original coordinates using the jacobian matrix

    dSdx = (j22[..., tf.newaxis] * shape_fn_grad[0]-j12[..., tf.newaxis
    ] * shape_fn_grad[1]) / jacobian_det[..., tf.newaxis]
    
    dSdy = (-j21[..., tf.newaxis] * shape_fn_grad[0]+j11[..., tf.newaxis
    ] * shape_fn_grad[1]) / jacobian_det[..., tf.newaxis]
    
    print("dSdx:\n"+str(dSdx)+"\n\ndSdy:"+str(dSdy)+"\n")

    pushfwd_shape_fn_grad = tf.concat((dSdx[tf.newaxis, ...],
    dSdy[tf.newaxis, ...]), axis=0)

    print("pushfwd_shape_fn_grad:\n"+str(pushfwd_shape_fn_grad)+"\n")

    # Computes the inverse of the jacobian

    J_inv_11 = j22/jacobian_det

    J_inv_12 = -j12/jacobian_det

    J_inv_21 = -j21/jacobian_det

    J_inv_22 = j11/jacobian_det

    J_row_1 = tf.stack([J_inv_11, J_inv_12], axis=-1)

    J_row_2 = tf.stack([J_inv_21, J_inv_22], axis=-1)

    print("J_inv_row_1:\n"+str(J_row_1)+"\n\nJ_inv_row_2:\n"+str(J_row_2
    )+"\n\n")

    J_inv = tf.stack([J_row_1, J_row_2], axis=-2)

    print("J_inv:\n"+str(J_inv)+"\n\n")

def stack_versus_concat():

    j11 = tf.constant([[1,2,3,4],[5,6,7,8]])

    j12 = tf.constant([[9,10,11,12],[13,14,15,16]])

    j21 = tf.constant([[17,18,19,20],[21,22,23,24]])

    j22 = tf.constant([[25,26,27,28],[29,30,31,32]])

    jacobian_det = j11*j22 - j12*j21

    J_inv_11 = j22/jacobian_det

    J_inv_12 = -j12/jacobian_det

    J_inv_21 = -j21/jacobian_det

    J_inv_22 = j11/jacobian_det

    print("J_inv_11:\n"+str(J_inv_11)+"\n\nJ_inv_12:\n"+str(J_inv_12)+
    "\n\nJ_inv_21:\n"+str(J_inv_21)+"\n\nJ_inv_22:\n"+str(J_inv_22)+"\n\n")

    J_inv_row_1 = tf.stack([J_inv_11, J_inv_12], axis=-1)

    J_inv_row_2 = tf.stack([J_inv_21, J_inv_22], axis=-1)

    J_inv = tf.stack([J_inv_row_1, J_inv_row_2], axis=-2)

    #print("J_inv_row_1:\n"+str(J_inv_row_1)+"\n\n")

    print("J_inv:\n"+str(J_inv)+"\n\n")

    dN_base = tf.constant([[1.,2.], [4.,5.], [7.,8.], [10.,11.]], dtype=
    J_inv.dtype)

    dN = tf.stack([dN_base, 2*dN_base, 3*dN_base], axis=-1)

    print("dN:\n"+str(dN)+"\n\n")

    dfdx = tf.einsum('eqij,qjk->eqik', J_inv, dN)

    print("The derivatives in the original system of coordinates:\n"+str(
    dfdx)+"\n\n")

def test_strain_energy():

    F = tf.constant([[[[1.0, 2.0, 0.0], [1.5, 2.0, 0.0], [0.0, 0.0, -1.0]],
    [[1.0, 2.0, 0.0], [1.5, 1.5, 0.0], [0.0, 0.0, 0.0]]],[[[1.0, 2.0, 0.0
    ], [1.5, 2.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 2.0, 0.0], [1.5, 2.0, 
    0.0], [0.0, 0.0, 0.0]]]])

    C = tf.matmul(F, F, transpose_a=True)

    print("\n\nright Cauchy-Green C:\n"+str(C))

    I1_C = tf.linalg.trace(C)

    J  = tf.linalg.det(F)

    ln_J = tf.math.log(J)

    print("The invariants of C are:\n\nI1=\n"+str(I1_C)+"\n\nJ=\n"+str(J
    )+"\n\nln(J)=\n"+str(ln_J)+"\n")

def test_gather_tensor_from_vector():

    u = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0])

    indices = tf.constant([[[0,1,2], [3,4,5], [6,7,8]], [[3,4,5], [6,7,8
    ], [9,10,11]]])

    u_elements = tf.gather(u, indices)

    print("The tensor of gathered DOFs of u field per element is:\n"+str(
    u_elements)+"\n")

def test_scatter_nd():

    n_global_dofs = 12

    global_residual_vector = tf.zeros([n_global_dofs], dtype=tf.float32)

    indices = tf.constant([[[0,1,2], [3,4,5], [7,6,8]]])

    u = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0])

    updates = tf.gather(u, indices)

    print("Updates:\n"+str(updates)+"\nShape updates at the third index:\n"+str(
    updates.shape[2])+"\n")

    # Adds the new dimension

    indices = tf.expand_dims(indices, axis=-1)

    print("New indices: "+str(indices)+"\n")

    global_residual_vector = tf.tensor_scatter_nd_add(global_residual_vector,
    indices, updates)

    print("The updated global vector is:\n"+str(global_residual_vector))

    ### Using variable

    global_residual_vector = tf.Variable(tf.zeros([n_global_dofs], dtype=tf.float32))

    global_residual_vector.scatter_nd_add(indices, updates)

    print("The updated global using variable is:\n"+str(global_residual_vector))

def test_pick_first_node():

    nodes = tf.constant([[[2.0, 0.5, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 
    0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]])

    x = nodes[...,0]

    y = nodes[...,1]

    z = nodes[...,2]

    print("\nx:\n"+str(x)+"\n\ny:"+str(y)+"\n")

    print("\nx of the first node:\n"+str(x[...,0])+"\n\ny of the first n"+
    "ode:\n"+str(y[...,0])+"\n")

    x -= x[:,2:3]

    y -= y[:,2:3]

    z -= z[:,2:3]

    print("The translated x coordinates are: "+str(x)+"\n")

    print("The translated y coordinates are: "+str(y)+"\n")

    print("The original coordinates:\n"+str(nodes)+"\n")

    vectors = tf.stack([x,y, z], axis=-1)

    print("The vectors are:\n"+str(vectors)+"\n")

    e1 = vectors[:,0,:]

    old_e2 = vectors[:,1,:]

    e3 = tf.linalg.cross(e1, old_e2)

    print("e1:\n"+str(e1)+"\n")

    print("e2:\n"+str(old_e2)+"\n")

    print("e3:\n"+str(e3)+"\n")

    norm_e1 = tf.linalg.norm(e1, axis=-1, keepdims=True)

    print("norm of e1:\n"+str(norm_e1)+"\n")

    norm_e3 = tf.linalg.norm(e3, axis=-1, keepdims=True)

    print("norm of e3:\n"+str(norm_e3)+"\n")

    e1 = e1/norm_e1

    e3 = e3/norm_e3

    print("Normalized e1:\n"+str(e1)+"\n")

    print("Normalized e3:\n"+str(e3)+"\n")

    e2 = tf.linalg.cross(e3, e1)

    print("Normalized new e2:\n"+str(e2)+"\n")

    project_old_e2_onto_e2 = tf.reduce_sum(old_e2*e2, axis=-1)

    print("Old e2 projected onto the direction of new e2:\n"+str(
    project_old_e2_onto_e2)+"\n")

    project_old_e2_onto_e1 = tf.reduce_sum(old_e2*e1, axis=-1)

    print("Old e2 projected onto the direction of e1:\n"+str(
    project_old_e2_onto_e1)+"\n")

    # Projects all vectors onto e1 to get the new x coordinates

    new_x = tf.stack([tf.reduce_sum(vectors[:,i,:]*e1, axis=-1) for (i
    ) in range(vectors.shape[1])], axis=-1)

    print("New x coordinates:\n"+str(new_x)+"\n")

    new_y = tf.stack([tf.reduce_sum(vectors[:,i,:]*e2, axis=-1) for (i
    ) in range(vectors.shape[1])], axis=-1)

    print("New y coordinates:\n"+str(new_y)+"\n")

if __name__=="__main__":

    test_gather_vector()

    test_gather_tensor()

    test_last_index()

    test_shape_functions_in_natural_coordinates()

    #get_tetra_coeffs()

    stack_versus_concat()

    get_quadrature_points()

    test_strain_energy()

    test_gather_tensor_from_vector()

    test_scatter_nd()

    test_pick_first_node()