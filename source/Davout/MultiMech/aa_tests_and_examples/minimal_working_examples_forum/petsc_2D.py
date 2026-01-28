from dolfin import *
from petsc4py import PETSc

# ------------------------------------------------------------------
# Mesh and function spaces
# ------------------------------------------------------------------

mesh = UnitSquareMesh(16, 16)

V = VectorElement("CG", mesh.ufl_cell(), 2)   # displacement
Q = FiniteElement("CG", mesh.ufl_cell(), 1)   # pressure

W = FunctionSpace(mesh, MixedElement([V, Q]))

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------

def left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

zero = Constant((0.0, 0.0))
bc_left = DirichletBC(W.sub(0), zero, left)

ux = Constant((1.0, 0.0))
bc_right = DirichletBC(W.sub(0), ux, right)


bcs = [bc_left, bc_right]

bc_p = DirichletBC(
    W.sub(1), Constant(0.0), left, method="pointwise"
)
bcs.append(bc_p)

# ------------------------------------------------------------------
# Unknowns and test functions
# ------------------------------------------------------------------

w = Function(W)
(u, p) = split(w)

(v, q) = TestFunctions(W)

dw = TrialFunction(W)

# ------------------------------------------------------------------
# Material parameters
# ------------------------------------------------------------------

mu = Constant(1.0)

I = Identity(2)
F = I + grad(u)
C = F.T * F
J = det(F)

# ------------------------------------------------------------------
# Incompressible Neo-Hookean energy
# ------------------------------------------------------------------

psi = (mu / 2.0) * (tr(C) - 2) - p * (J - 1)

Pi = psi * dx

# ------------------------------------------------------------------
# Residual and Jacobian
# ------------------------------------------------------------------

R = derivative(Pi, w, TestFunction(W))
J_form = derivative(R, w, dw)

# ------------------------------------------------------------------
# PETSc options (CRUCIAL PART)
# ------------------------------------------------------------------

#"""
opts = PETSc.Options()

# Nonlinear solver
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "bt"
opts["snes_rtol"] = 1e-6
opts["snes_atol"] = 1e-8
opts["snes_monitor"] = None

# Linear solver
opts["ksp_type"] = "gmres"
opts["ksp_rtol"] = 1e-6
opts["ksp_atol"] = 1e-8
opts["ksp_max_it"] = 500
opts["ksp_monitor"] = None

# FieldSplit (Schur complement)
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "schur"
opts["pc_fieldsplit_schur_factorization_type"] = "lower"
opts["pc_fieldsplit_schur_precondition"] = "selfp"

# Displacement block
opts["fieldsplit_0_ksp_type"] = "cg"
opts["fieldsplit_0_pc_type"] = "hypre"

# Pressure block
opts["fieldsplit_1_ksp_type"] = "preonly"
opts["fieldsplit_1_pc_type"] = "jacobi"
opts["ksp_view"] = None
opts["snes_view"] = None

#""

# ------------------------------------------------------------------
# Nonlinear problem
# ------------------------------------------------------------------

problem = NonlinearVariationalProblem(R, w, bcs, J_form)
solver = NonlinearVariationalSolver(problem)

solver.parameters["nonlinear_solver"] = "snes"

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------

solver.solve()

# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

(u_sol, p_sol) = w.split()

File("u.pvd") << u_sol
File("p.pvd") << p_sol
