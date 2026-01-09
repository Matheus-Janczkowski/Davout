from fenics import *
import ufl_legacy as ufl
from dolfin import *

from Grante.MultiMech.tool_box import mesh_handling_tools as mesh_tools

from Grante.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

def coordinates_direction(circumferencial_direction_file_name, axial_direction_file_name, radial_direction_file_name, mesh_file_name, parent_path):

    #     |-------------------------------------------------------------------------|
    #     |   This code aims to apply a simple exemple of a heat transfer exercise  |
    #     |               a plane wall to validate the FeNiCs prompt                |
    #     |-------------------------------------------------------------------------|

    # Defines the geometry mesh

    quadrature_degree = 2

    mesh_fileName = mesh_file_name

    verbose = False

    mesh_data_class = mesh_tools.read_mshMesh(mesh_fileName, 
        quadrature_degree=quadrature_degree, verbose=verbose)

    # First, it is necessary to define the goemtry and the mesh (Box mesh)

    polynomial_degree = 1

    # Defines the prescribed temperatues conditions 

    T_hot = 600.0
    T_cold = 400.0

    # We are considering a total isotropic material, thus, the sencond order tensor 
    # of the condutivity is gonna be k*I, where I is the identity tensor 

    # ************************************************************
    #                     Axial and radial heat transfer         *
    # ************************************************************

    k = 100.0

    I = Identity(3)

    k_tensor = k*I

    # Defines the function Space

    V = FunctionSpace(mesh_data_class.mesh, 'P', polynomial_degree)

    # Defines the Dirichlet boundary conditions for axial transfer heat flux 

    bc_hot_axial = DirichletBC(V, Constant(T_hot), mesh_data_class.boundary_meshFunction, mesh_data_class.boundary_physicalGroupsNameToTag["bottom"])
    bc_cold_axial = DirichletBC(V, Constant(T_cold), mesh_data_class.boundary_meshFunction, mesh_data_class.boundary_physicalGroupsNameToTag["top"])

    # Defines the Dirichlet boundary conditions for radial transfer heat flux 

    bc_cold_radial = DirichletBC(V, Constant(T_cold), mesh_data_class.boundary_meshFunction, mesh_data_class.boundary_physicalGroupsNameToTag["lateral external"])

    # Put in a list

    bc_axial = [bc_hot_axial, bc_cold_axial]
    bc_radial = [bc_cold_radial]

    # ************************************************************
    #                         variational Form                   *
    # ************************************************************

    T = TrialFunction(V)

    delta_T = TestFunction(V)

    # To test, first, we can define the energy generation as constant zero

    q_v_annulus = Constant(0.0)
    q_v_nucleo = Constant(300.0)

    # Recording coersivity, as shown by Larx Milgran, the bilinear and linear parts are:
    # We have only one volume, thus:

    dx = mesh_data_class.dx

    # bilinear part

    a = dot(k_tensor*grad(T), grad(delta_T)) * dx

    # linear part (if had heat flux prescribed, it would be necessary to apply more one part: 
    # (heat_flux * delta_T * ds(number of the face)
    # for the axial direction prompt code isn't used the generation energy, just only in the 
    # radial direction prompt code

    l = (q_v_annulus * delta_T * dx(mesh_data_class.domain_physicalGroupsNameToTag[
    "annulus"])) + (q_v_nucleo * delta_T * dx(mesh_data_class.domain_physicalGroupsNameToTag[
    "nucleus"]))

    # Solve 

    T_solve_axial = Function(V, name = "Temperature axial")
    T_solve_radial = Function(V, name = "Temperature radial")

    solve(a == l, T_solve_radial, bc_radial)

    write_field_to_xdmf({"monolithic solution": T_solve_radial, "mesh file": mesh_fileName}, directory_path= 
    parent_path, visualization_copy=True, explicit_file_name="temperature_radial")

    q_v_nucleo.assign(0.0)

    solve(a == l, T_solve_axial, bc_axial)

    # ************************************************************
    #                 Projection of the heat_flux                *
    # ************************************************************

    W = VectorFunctionSpace(mesh_data_class.mesh, 'P', 1)

    # We know that: heat_flux = -K * grad(T)

    direction_axial = -1.0 * k_tensor * grad(T_solve_axial)
    direction_radial = -1.0 * k_tensor * grad(T_solve_radial)

    direction_norm_axial = sqrt(dot(direction_axial, direction_axial))
    direction_norm_radial = sqrt(dot(direction_radial, direction_radial))

    modulator_axial = conditional(gt(direction_norm_axial, 0.0), 1/direction_norm_axial, 1.0)
    modulator_radial = conditional(gt(direction_norm_radial, 0.0), 1/direction_norm_radial, 1.0)

    e_axialFunction = modulator_axial*direction_axial
    e_radialFunction = modulator_radial*direction_radial

    # cg is a iterative method

    e_radial = project(e_radialFunction, W, solver_type="cg")

    e_radial.rename("Radial direction", "Projection")

    e_axial = project(e_axialFunction, W, solver_type="cg")

    e_axial.rename("a direction", "Projection")

    e_circumferential = project(cross(e_radial, e_axial), W, solver_type="cg")

    e_circumferential.rename("d direction", "Projection")

    # ************************************************************
    #                         Save in a file                     *
    # ************************************************************

    parent_path = parent_path

    field_file_name_circumferential = circumferencial_direction_file_name

    field_file_name_axial = axial_direction_file_name

    field_file_name_radial = radial_direction_file_name

    write_field_to_xdmf({"monolithic solution": e_circumferential,
    "mesh file": mesh_fileName}, directory_path= 
    parent_path, visualization_copy=True, explicit_file_name=field_file_name_circumferential)

    write_field_to_xdmf({"monolithic solution": e_axial, "mesh file": mesh_fileName}, directory_path= 
    parent_path, visualization_copy=True, explicit_file_name=field_file_name_axial)

    write_field_to_xdmf({"monolithic solution": e_radial, "mesh file": mesh_fileName}, directory_path= 
    parent_path, visualization_copy=True, explicit_file_name=field_file_name_radial)

if __name__=="__main__":

    mesh_file_name = ("/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/Interverteb"+
    "ralDiscANN/material_properties_interpolation/intervertebral_disc_mesh")

    parent_path = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation"

    circumferential_direction_file_name = "circumferential_direction_000"

    axial_direction_file_name = "axial_direction_000"

    radial_direction_file_name = "radial_direction_000"

    coordinates_direction(circumferential_direction_file_name, axial_direction_file_name, radial_direction_file_name, mesh_file_name, parent_path)