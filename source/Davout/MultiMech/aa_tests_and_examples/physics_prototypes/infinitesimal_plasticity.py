# Routine to experiment with quadrature elements to allow for implemen-
# tations concerned with algorithmic procedures in the quadrature points

from dolfin import *

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Davout.MultiMech.constitutive_models.plasticity.isotropic_infinitesimal_plasticity import ExponentialHardeningElastoPlasticWithBackStress

from .....Davout.MultiMech.constitutive_models.hyperelasticity.infinitesimal_elasticity import LinearIsotropicElastic

from .....Davout.MultiMech.tool_box import functional_tools

from .....Davout.MultiMech.tool_box import variational_tools

from .....Davout.MultiMech.tool_box.newton_raphson_tools import NewtonRaphson

from .....Davout.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

# Creates a box mesh using gmsh

shape_function_degree = 2

quadrature_degree = int(2*(shape_function_degree-1))

mesh_data_class = read_mshMesh({"length x": 0.3, "length y": 0.2, "len"+
"gth z": 3.0, "number of divisions in x": 5, "number of divisions in y":
5, "number of divisions in z": 40, "verbose": False, "mesh file name": 
"box_mesh", "mesh file directory": get_parent_path_of_file()}, 
quadrature_degree=quadrature_degree)

# Neumann boundary conditions

maximum_load = 2.2E3

plastic_material = True

# Sets the number of load steps (pseudo-time steps)

n_load_steps = 5

# Sets the constitutive model

E = 1E6

poisson = 0.3

# Sets the coefficients of the plastic behavior

sigma_y = 2E3

H_back_stress = 0.0

K_1 = E*0.02

K_2 = 0.0

theta = 0.0

constitutive_models = None 

if plastic_material:

    constitutive_models = {"volume 1": 
    ExponentialHardeningElastoPlasticWithBackStress({"E": E, "nu": 
    poisson, "sigma_y": sigma_y, "H": H_back_stress, "K_1": K_1, "K_2": 
    K_2, "theta": theta}, newton_raphson_tolerance=1E-4)}

else:

    constitutive_models = {"volume 1": LinearIsotropicElastic({"E": E, 
    "nu": poisson})}

# Checks the constitutive models and ensures that the supplied param-
# eters are correct

for constitutive_model in constitutive_models.values():

    constitutive_model.check_model({"mesh_data_class": mesh_data_class})

# Sets the function space

functional_data_class = functional_tools.construct_monolithicFunctionSpace(
{"Displacement": {"field type": "vector", "interpolation function": "C"+
"G", "polynomial degree": shape_function_degree}}, mesh_data_class, 
verbose=True)

# Dirichlet boundary conditions

(bc, dirichlet_loads, functional_data_class, elements_dictionary, 
variational_form, post_processes, post_processesSubmesh, solution_name) = (
functional_tools.construct_DirichletBCs({"bottom": {"BC case": "Simple"+
"SupportDirichletBC", "bc_informationsDict": {"list_of_constrained_dofs": 
[2]}}, "right": {"BC case": "SimpleSupportDirichletBC", "bc_information"+
"sDict": {"list_of_constrained_dofs": [1]}}, "back": {"BC case": "Simp"+
"leSupportDirichletBC", "bc_informationsDict": {"list_of_constrained_d"+
"ofs": [0]}}}, functional_data_class, mesh_data_class))

# Variational form of the exterior work using an uniform referential 
# traction

external_work, neumann_loads = variational_tools.traction_work({"top": {
"load case": "UniformReferentialTraction", "amplitude_tractionX": 
0.0, "amplitude_tractionY": 0.0, "amplitude_tractionZ": maximum_load, 
"parametric_load_curve": "linear", "t": 0.0, "t_final": 1.0}}, 
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

    print("\nUpdates the constitutive model at '"+str(physical_group)+
    "'")

    # Updates the internal variables and the tangent modulus. The second
    # argument is false, because this update is not inside a Newton-
    # Raphson iteration

    if plastic_material:

        constitutive_model.update_internal_variables(
        functional_data_class.solution_fields["Displacement"], 
        flag_newton_raphson_iteration=False)

    # Gets the Cauchy stress tensor from the linear elastic constitutive
    # relation and pulls it back to the first Piola-Kirchhoff stress 
    # tensor. This way, a formulation of large displacements and infini-
    # tesimal strains can be accomplished, since the linearization of 
    # the first Piola-Kirchhoff stress tensor will carry the geometric
    # nonlinearities

    sigma = constitutive_model.cauchy_stress(
    functional_data_class.solution_fields["Displacement"])

    P = J*sigma*F_invT

    internal_work += (inner(P, (grad(
    functional_data_class.variation_fields["Displacement"])))*
    mesh_data_class.dx(mesh_data_class.domain_physicalGroupsNameToTag[
    physical_group]))

# Defines a function to update the variational form

verbose = True

def variational_form_update_function(pseudo_time, 
flag_newton_raphson_iteration, verbose=verbose):

    # Update the load class
    
    neumann_loads[0].update_load(pseudo_time, verbose=verbose)

    if verbose:

        print("\nUpdates the constitutive model at '"+str(physical_group
        )+"'")

    # Updates the internal variables and the tangent modulus

    if plastic_material:

        constitutive_models["volume 1"].update_internal_variables(
        functional_data_class.solution_fields["Displacement"],
        flag_newton_raphson_iteration=flag_newton_raphson_iteration, 
        verbose=verbose)

# Defines a class to perform post-processes. In this case, simply sa-
# ving the output field

class PostProcesses:

    def __init__(self, visualization_copy):

        self.xdmf_file = None

        self.visualization_copy_file = None

        self.visualization_copy = visualization_copy

    def __call__(self, pseudo_time, time_step):

        # Saves this pseudo-time step in a xdmf file

        if self.visualization_copy:

            self.xdmf_file, self.visualization_copy_file = write_field_to_xdmf(
            functional_data_class, explicit_file_name=
            get_parent_path_of_file()+"//displacement_finite_deformat"+
            "ion_with_infinitesimal_strain.xdmf", file=self.xdmf_file, 
            time=pseudo_time, time_step=time_step, visualization_copy=
            self.visualization_copy, visualization_copy_file=
            self.visualization_copy_file,
            flag_read_copy=True, close_file=False)

        else:

            self.xdmf_file = write_field_to_xdmf(functional_data_class, 
            explicit_file_name=get_parent_path_of_file()+"//displaceme"+
            "nt_finite_deformation_with_infinitesimal_strain.xdmf", 
            file=self.xdmf_file, time=pseudo_time, time_step=time_step)

post_processes = PostProcesses(True)
    
# Sets the Newton-Raphson solver

newton_raphson_class = NewtonRaphson(n_load_steps, variational_form=(
internal_work-external_work), functional_data_class=
functional_data_class, boundary_conditions_list=bc, 
variational_form_update_function=variational_form_update_function,
post_processing_function=post_processes, use_custom_newton=True)

# Calls the loading procedure

newton_raphson_class()

if plastic_material:

    print("\nEvaluates the internal variables after the full load hist"+
    "ory")

    constitutive_models["volume 1"].update_internal_variables(
    functional_data_class.solution_fields["Displacement"], 
    flag_newton_raphson_iteration=False, verbose=True)