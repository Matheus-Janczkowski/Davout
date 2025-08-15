# Routine to test a hyperelastic disc

import os

from dolfin import *

import ufl_legacy as ufl

from mshr import *

import numpy as np

import source.tool_box.mesh_handling_tools as mesh_tools

import source.tool_box.pseudotime_stepping_tools as newton_tools

import source.tool_box.file_handling_tools as file_tools

import tests.test_meshes.beam_gmsh as beam_gmsh

def test():

    ########################################################################
    ########################################################################
    ##                      User defined parameters                       ##
    ########################################################################
    ########################################################################

    ########################################################################
    #                          Simulation results                          #
    ########################################################################

    # Defines the path to the results directory 

    results_pathGraphics = (os.getcwd()+"//tests//micropolar//Bauer_et_al/"+
    "/results//graphics")

    displacement_fileName = "displacement_neo_hookean.xdmf"

    ########################################################################
    #                         Material properties                          #
    ########################################################################

    # Sets a dictionary of properties

    material_properties = dict()

    mu = 26.12

    K_constitutive = 63.84

    lmbda = K_constitutive-(2*mu/3)

    ########################################################################
    #                                 Mesh                                 #
    ########################################################################

    # Defines the name of the file to save the mesh in. Do not write the fi-
    # le termination, e.g. .msh or .xdmf; both options will be saved automa-
    # tically

    mesh_fileName = "tests//test_meshes//micropolar_beam"

    # Generates the mesh

    ratio_Lb = 1.5E-1

    gamma = 1.18E0

    beta = 0.0

    n_volumes = 1

    beam_gmsh.generate_micropolarBeam(mu, ratio_Lb, beta, gamma, 
    mesh_fileName, n_volumes, transfinite=False)

    (mesh, dx, ds, n, domain_meshCollection, domain_meshFunction, 
    boundary_meshCollection, boundary_meshFunction,
    domain_physGroupsNamesToTags, boundary_physGroupsNamesToTags
    ) = mesh_tools.read_mshMesh(mesh_fileName)

    """########################################################################
    #                                 Mesh                                 #
    ########################################################################

    # Defines the name of the file to save the mesh in. Do not write the fi-
    # le termination, e.g. .msh or .xdmf; both options will be saved automa-
    # tically

    file_name = "Meshes//cylinder_sym"#"Meshes//periodic_beam"

    #file_name = "Meshes//periodic_beam"

    # Defines a flag to generate a new mesh or not

    flag_newMesh = True

    # Generates the mesh and writes it 

    if flag_newMesh:

        mesher.generate_periodicMesh(file_name, flag_transfinite=1, verbose=
        True)

    # Initializes the mesh object and reads the xdmf file

    mesh = Mesh()

    # Initializes a mesh value collection to store mesh data. Uses the topo-
    # logy dimension given by the mesh proper

    domain_meshCollection = MeshValueCollection("size_t", mesh, 
    mesh.topology().dim())

    # Reads the mesh with domain physical groups

    with XDMFFile(file_name+"_domain.xdmf") as infile:

        infile.read(mesh)

        infile.read(domain_meshCollection, "domain")

    # Initializes a mesh value collection to store mesh data of the bounda-
    # ries. Lessens the topology dimension by a factor of 1

    boundary_meshCollection = MeshValueCollection("size_t", mesh, 
    mesh.topology().dim()-1)

    # Reads the mesh with surface physical groups

    with XDMFFile(file_name+"_boundary.xdmf") as infile:

        infile.read(boundary_meshCollection, "boundary")

    # Converts the mesh value collections to a mesh functions, for mesh va-
    # lue collections are low level and cannot be used for FEM integration
    # and other higher level operations inside FEniCs

    cell_markers = cpp.mesh.MeshFunctionSizet(mesh, domain_meshCollection)

    facet_markers = cpp.mesh.MeshFunctionSizet(mesh, boundary_meshCollection)

    # Createsa copy of the cell markers to use in the submesh

    submesh_cellMarkers = cpp.mesh.MeshFunctionSizet(mesh, 
    domain_meshCollection)"""

    ########################################################################
    #                           Function spaces                            #
    ########################################################################

    # Defines the interpolation function and its degree

    interpolation_function = "Lagrange"

    element_degree = 2

    n_gaussPoints = 3

    # Defines a vectorial function space

    V = VectorFunctionSpace(mesh, interpolation_function, degree=
    element_degree)

    # Defines a trial function

    Delta_u = TrialFunction(V)

    u = Function(V)

    # Defines a test function

    du = TestFunction(V)

    ########################################################################
    #                          Integral measures                           #
    ########################################################################

    # Defines the integration measures using the mesh information retrieved
    # from GMSH

    dx = Measure("dx", domain=mesh, subdomain_data=domain_meshFunction, metadata={
    "quadrature_degree": n_gaussPoints})

    ds = Measure("ds", domain=mesh, subdomain_data=boundary_meshFunction, metadata={
    "quadrature_degree": n_gaussPoints})

    ########################################################################
    #                          Pseudotime control                          #
    ########################################################################

    t = 0.0

    t_final = 1.0

    n_steps = 20

    delta_t = t_final/n_steps

    ########################################################################
    #                     Dirichlet boundary conditions                    #
    ########################################################################

    # Defines the expressions for the Dirichlet boundary conditions

    expression_bottom = Constant("0.0")

    expression_aft = Constant("0.0")

    expression_starboard = Constant("0.0")

    expression_cantilever = Constant(("0.0", "0.0", "0.0"))

    # Creates a list of boundary conditions using the expressions above, the
    # mesh function with the facet information, and the desired surface phy-
    # sical group where upon the boundary condition is to be applied

    #bc_dirichlet = [DirichletBC(V.sub(2), expression_bottom,facet_markers,5),
    #DirichletBC(V.sub(0), expression_aft, facet_markers, 8),
    #DirichletBC(V.sub(1), expression_starboard, facet_markers, 9)]

    bc_dirichlet = [DirichletBC(V, expression_cantilever,boundary_meshFunction,2)]

    ########################################################################
    #                      Neumann boundary conditions                     #
    ########################################################################

    # Defines the traction vector

    T = Expression(("0.0", "(t/t_final)*T_max", "0.0"), t_final=t_final,
    t=0.0, T_max=4E0, degree=0) 

    # Defines the body forces vector

    B = Constant(("0.0","0.0","0.0"))

    ########################################################################
    #                              Tensorial                               #
    ########################################################################

    I = Identity(3)

    # Defines the deformation gradient

    F = grad(u)+I 

    # Defines the Euler-Lagrange strain tensor

    C = variable((F.T)*F) 

    # Defines the Saint Venant energy function

    #psi = (mu*inner(E,E))+(0.5*lmbda*(tr(E)**2))

    # Defines the Neo-Hookean energy function

    I1_C = tr(C)

    I2_C = det(C)

    J = sqrt(I2_C)
            
    # Evaluates the trace-related part

    psi = (mu/2)*(I1_C-3)-(mu*ln(J))+((lmbda*0.5)*((ln(J))**2))

    # Defines the second Piola-Kirchhof stress tensor

    S = 2*diff(psi, C)

    # Defines the first Piola-Kirchhof stress tensor

    P = F*S

    dF = grad(du)

    ########################################################################
    #                           Weak formulation                           #
    ########################################################################
                            
    # Defines the bilinear form

    a = inner(P,dF)*dx(1)

    # Defines the linear form

    L = (dot(B,du)*dx)+(dot(T,du)*ds(4))

    residue = a-L

    Jacobian = derivative(residue, u, Delta_u)

    problem = NonlinearVariationalProblem(residue, u, bc_dirichlet, J=Jacobian)

    solution = NonlinearVariationalSolver(problem)

    solution.parameters["nonlinear_solver"] = "newton"

    solution.parameters["newton_solver"]["absolute_tolerance"] = 1E-8

    solution.parameters["newton_solver"]["relative_tolerance"] = 1E-7

    solution.parameters["newton_solver"]["maximum_iterations"] = 40

    solution.parameters["newton_solver"]["linear_solver"] = "mumps"

    # Sweeps through the pseudotime steps

    reaction_z = 0.0

    # Creates the displacement file

    file = XDMFFile(results_pathGraphics+"//"+displacement_fileName)

    for i in range(n_steps):

        # Updates time and the traction vector

        t += delta_t

        print("\nRuns pseudotime", i+1, "of", n_steps, ":", t, "\n")

        T.t = t

        # Solves the system

        solution.solve()

        # = as_vector([0.0,0.0,1.0])

        #reaction_z = assemble((dot(N,P*N))*ds)

        #print("\nThe reaction in the z direction is:",
        #reaction_z, "\n")

        file.write(u, t)

test()