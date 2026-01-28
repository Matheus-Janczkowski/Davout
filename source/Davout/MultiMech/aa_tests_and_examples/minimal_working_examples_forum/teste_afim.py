from dolfin import *
import matplotlib.pyplot as plt

# Create a 3D mesh (box)
x_min, y_min, z_min = 0.0, 0.0, 0.0
x_max, y_max, z_max = 1.0, 1.0, 1.0
mesh = BoxMesh(Point(x_min, y_min, z_min), Point(x_max, y_max, z_max), 20, 20, 20)

# Define the function space for the vector field
V = VectorFunctionSpace(mesh, 'CG', 1)

# Define the affine field parameters
u0 = Constant((0.1, 0.2, 0.3))  # A constant vector
G = Constant(((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3)))  # A tensor
xc = Constant((0.5, 0.5, 0.5))  # Centroid

# Define the affine field using a UserExpression
class AffineField(UserExpression):
    def __init__(self, u0, G, xc, **kwargs):
        super().__init__(**kwargs)
        self.u0 = u0
        self.G = G
        self.xc = xc

    def eval(self, value, x):

        u0_values = self.u0.values()
        G_values = self.G.values()

        G_matrix = []

        for i in range(3):
            G_matrix.append([])
            for j in range(3):

                G_matrix[-1].append(G_values[int((i*3)+j)])

        x_vec = [x[0]-self.xc.values()[0], x[1]-self.xc.values()[1], x[2]-self.xc.values()[2]]
        value[0] = self.u0.values()[0]+(x_vec[0]*G_matrix[0][0])+(x_vec[1]*G_matrix[0][1])+(x_vec[2]*G_matrix[0][2])
        value[1] = self.u0.values()[1]+(x_vec[0]*G_matrix[1][0])+(x_vec[1]*G_matrix[1][1])+(x_vec[2]*G_matrix[1][2])
        value[2] = self.u0.values()[2]+(x_vec[0]*G_matrix[2][0])+(x_vec[1]*G_matrix[2][1])+(x_vec[2]*G_matrix[2][2])

    def value_shape(self):
        return (3,)

# Create an instance of the affine field
affine_field_expr = AffineField(u0=u0, G=G, xc=xc, degree=1)

# Project the affine field to the function space for visualization
affine_field = interpolate(affine_field_expr, V)

print(affine_field([0.5, 0.5, 0.5]))

print(affine_field([0.0, 0.0, 0.0]))

print(affine_field([1.0, 1.0, 1.0]))

print("\n\n\n")
u0.assign(Constant((0.2, 0.1, 0.5)))  # A constant vector
G.assign(Constant(((0.2, 0.0, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, 0.4)))) 

print(affine_field([0.5, 0.5, 0.5]))

print(affine_field([0.0, 0.0, 0.0]))

print(affine_field([1.0, 1.0, 1.0]))

# Plotting the field (magnitude)
file = XDMFFile("teste_afim.xdmf")

file.write(affine_field)