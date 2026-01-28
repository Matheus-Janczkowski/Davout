# Routine to experiment with incompressible hyperelasticity

from dolfin import *

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Grante.MultiMech.constitutive_models.hyperelasticity.isotropic_hyperelasticity import Neo_Hookean

from .....Grante.MultiMech.tool_box import functional_tools

from .....Grante.MultiMech.tool_box import variational_tools

from .....Grante.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

# Creates a box mesh using gmsh

mesh_data_class = read_mshMesh({"length x": 0.3, "length y": 0.2, "len"+
"gth z": 1.0, "number of divisions in x": 5, "number of divisions in y":
5, "number of divisions in z": 25, "verbose": False, "mesh file name": 
"box_mesh", "mesh file directory": get_parent_path_of_file()})

# Neumann boundary conditions

maximum_load = 5E5

# Sets the constitutive model

E = 1E6

poisson = 0.3

constitutive_models = {"volume 1": Neo_Hookean({"E": E, "nu": poisson})}

# Sets the function space

functional_data_class = functional_tools.construct_monolithicFunctionSpace(
{"Displacement": {"field type": "vector", "interpolation function": "C"+
"G", "polynomial degree": 2}, "Pressure": {"field type": "scalar", "in"+
"terpolation function": "CG", "polynomial degree": 1}}, mesh_data_class)

# Dirichlet boundary conditions

bc, dirichlet_loads = functional_tools.construct_DirichletBCs({"bottom": 
{"BC case": "FixedSupportDirichletBC", "sub_fieldsToApplyBC": "Displac"+
"ement"}}, functional_data_class, mesh_data_class)

# Variational form of the exterior work using an uniform referential 
# traction

external_work, neumann_loads = variational_tools.traction_work({"top": {
"load case": "UniformReferentialTraction", "amplitude_tractionX": 0.0, 
"amplitude_tractionY": 0.0, "amplitude_tractionZ": maximum_load, "para"+
"metric_load_curve": "square_root", "t": 0.0, "t_final": 1.0}}, "Displ"+
"acement", functional_data_class, mesh_data_class, [])

# Update the load class

neumann_loads[0].update_load(1.0)

# Gets the variational form of the inner work

internal_work = 0.0

for physical_group, constitutive_model in constitutive_models.items():

    internal_work += (inner(constitutive_model.first_piolaStress(
    functional_data_class.solution_fields["Displacement"]), grad(
    functional_data_class.variation_fields["Displacement"]))*
    mesh_data_class.dx(mesh_data_class.domain_physicalGroupsNameToTag[
    physical_group]))

# Solver

solver = functional_tools.set_nonlinearProblem((internal_work-
external_work), functional_data_class, bc)

solver.solve()

# Solution saving

write_field_to_xdmf(functional_data_class)