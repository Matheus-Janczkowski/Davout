# Routine to experiment with quadrature elements to allow for implemen-
# tations concerned with algorithmic procedures in the quadrature points

from dolfin import *

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ...tool_box.mesh_handling_tools import read_mshMesh

from ...constitutive_models.plasticity.isotropic_infinitesimal_plasticity import ExponentialHardeningElastoPlasticWithBackStress

from ...tool_box import functional_tools

from ...tool_box import variational_tools

from ...tool_box.projection_tools import MultipleFieldsProjectionOntoQuadratureSpace

from ...tool_box.read_write_tools import write_field_to_xdmf

# Creates a box mesh using gmsh

mesh_data_class = read_mshMesh({"length x": 0.3, "length y": 0.2, "len"+
"gth z": 1.0, "number of divisions in x": 5, "number of divisions in y":
5, "number of divisions in z": 25, "verbose": False, "mesh file name": 
"box_mesh", "mesh file directory": get_parent_path_of_file()})

# Neumann boundary conditions

maximum_load = 5E3

# Sets the number of load steps (pseudo-time steps)

n_load_steps = 5

# Sets the constitutive model

E = 1E6

poisson = 0.3

# Sets the coefficients of the plastic behavior

sigma_y =120E6

H_back_stress = 0.0

K_1 = E*0.02

K_2 = 0.0

theta = 0.0

constitutive_models = {"volume 1": 
ExponentialHardeningElastoPlasticWithBackStress({"E": E, "nu": poisson,
"sigma_y": sigma_y, "H": H_back_stress, "K_1": K_1, "K_2": K_2, "theta":
theta})}

# Checks the constitutive models and ensures that the supplied param-
# eters are correct

for constitutive_model in constitutive_models.values():

    constitutive_model.check_model({"mesh_data_class": mesh_data_class})

# Sets the function space

functional_data_class = functional_tools.construct_monolithicFunctionSpace(
{"Displacement": {"field type": "vector", "interpolation function": "C"+
"G", "polynomial degree": 2}}, mesh_data_class, verbose=True)

# Dirichlet boundary conditions

(bc, dirichlet_loads, functional_data_class, elements_dictionary, 
variational_form, post_processes, post_processesSubmesh, solution_name) = (
functional_tools.construct_DirichletBCs({"bottom": 
{"BC case": "FixedSupportDirichletBC"}}, functional_data_class, 
mesh_data_class))

# Variational form of the exterior work using an uniform referential 
# traction

external_work, neumann_loads = variational_tools.traction_work({"top": {
"load case": "UniformReferentialTraction", "amplitude_tractionX": 
maximum_load, "amplitude_tractionY": 0.0, "amplitude_tractionZ": 0.0, 
"parametric_load_curve": "square_root", "t": 0.0, "t_final": 1.0}}, 
"Displacement", functional_data_class, mesh_data_class, [])

# Computes the elements necessary to pull-back the Cauchy stress to the
# first Piola-Kirchhoff stress tensor using Nanson's formula

I = Identity(functional_data_class.solution_fields["Displacement"
].geometric_dimension())

F = grad(functional_data_class.solution_fields["Displacement"])+I

F_invT = inv(F).T

J = det(F)

# Gets the variational form of the inner work

internal_work = 0.0

for physical_group, constitutive_model in constitutive_models.items():

    # Updates the internal variables and the tangent modulus

    constitutive_model.update_internal_variables_and_tangent_modulus(
    functional_data_class.solution_fields["Displacement"])

    # Gets the Cauchy stress tensor from the linear elastic constitutive
    # relation and pulls it back to the first Piola-Kirchhoff stress 
    # tensor. This way, a formulation of large displacements and infini-
    # tesimal strains can be accomplished, since the linearization of 
    # the first Piola-Kirchhoff stress tensor will carry the geometric
    # nonlinearities

    sigma = constitutive_model.tangent_cauchy_stress(
    functional_data_class.solution_fields["Displacement"])

    P = J*sigma*F_invT

    internal_work += (inner(P, grad(
    functional_data_class.variation_fields["Displacement"]))*
    mesh_data_class.dx(mesh_data_class.domain_physicalGroupsNameToTag[
    physical_group]))
    
# Sets the Newton-Raphson solver

solver = functional_tools.set_nonlinearProblem((internal_work-
external_work), functional_data_class, bc)

# Initializes the file where the solution is going to be saved

xdmf_file = None

# Iterates over the load steps

for i in range(n_load_steps):

    print("\nStarts the "+str(i+1)+"-th load step")

    # Evaluates the current pseudo time

    pseudo_time = (i/(n_load_steps-1))

    # Update the load class

    neumann_loads[0].update_load(pseudo_time)

    # Calls the solver

    solver.solve()

    # Saves this pseudo-time step in a xdmf file

    xdmf_file = write_field_to_xdmf(functional_data_class, 
    explicit_file_name=get_parent_path_of_file()+"//displacement_finit"+
    "e_deformation_with_infinitesimal_strain.xdmf", file=xdmf_file, time
    =pseudo_time, time_step=i)