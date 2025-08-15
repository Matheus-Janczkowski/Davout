from dolfin import *
import numpy as np

import source.multiscale.multiscale_expressions as multiscale_expressions

L, H, W = 1.0, 0.2, 0.3
mesh = BoxMesh(Point(0,0,0), Point(W,H,L), 1,1,1)

lower_facet = CompiledSubDomain("near(x[1], 0)")
left_facet = CompiledSubDomain("near(x[2], 0)")
right_facet = CompiledSubDomain("near(x[2], L)", L=L)
volume_1 = CompiledSubDomain("x[2]<=L*0.5", L=L)
volume_2 = CompiledSubDomain("x[2]>L*0.5", L=L)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
#boundary_markers.set_all(0)
lower_facet.mark(boundary_markers, 2)
left_facet.mark(boundary_markers, 3)
right_facet.mark(boundary_markers, 6)

volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
#volume_markers.set_all(0)
volume_1.mark(volume_markers, 7)
volume_2.mark(volume_markers, 8)

class MeshData:

    def __init__(self, mesh_data, volume_markers, boundary_markers):

        self.mesh = mesh_data

        self.x = SpatialCoordinate(mesh_data)

        self.dx = Measure("dx", domain=self.mesh, subdomain_data=
        volume_markers, metadata={"quadrature_degree": 2})

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=
        boundary_markers)

mesh_dataClass = MeshData(mesh, volume_markers, boundary_markers)

volume_inverse = (1.0/assemble(1.0*mesh_dataClass.dx))

class Macro:

    def __init__(self):

        self.displacement = [0.0, 0.0, 0.0]

        self.displacement_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]

field_expression, field_correction, centroid = (
multiscale_expressions.construct_fieldCorrections(False, volume_inverse, 
mesh_dataClass, [Macro()], "displacement", "displacement_gradient", {
"displacement":VectorElement("CG", mesh.ufl_cell(), 2)}, None))

pbc = multiscale_expressions.PeriodicCubicBoundary(*centroid, 
mesh_dataClass)

V = VectorFunctionSpace(mesh, "Lagrange", 2, constrained_domain=pbc)

W = VectorFunctionSpace(mesh, "Lagrange", 1)

A = VectorFunctionSpace(mesh, "Lagrange", 2)

B = FunctionSpace(mesh, "Lagrange",2)

dofs = np.array(W.dofmap().dofs(mesh,0)).reshape(mesh.num_vertices(),3)

dof_coords = W.tabulate_dof_coordinates()

print(dof_coords, len(W.tabulate_dof_coordinates()))

print("")

i = 0

mesh_coords = mesh.coordinates()

for i in range(len(mesh_coords)):

    print(dofs[i], mesh_coords[i], " -> ", [dof_coords[j] for j in dofs[i]])

    i += 1

print(W.ufl_element().value_size())

print(np.array(B.dofmap().dofs(mesh,0)).reshape(mesh.num_vertices(),1))

print(dofs[1])

bc = DirichletBC(A, Constant(0.0), dofs[1], dofs[1][0], method="pointwise")