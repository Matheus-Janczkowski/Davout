from fenics import *

a = 10
## Create mesh and define function space
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(a, a, a), a, a, a)

## u_D(x,y) = 1 + x^2 + 2y^2
## Expression object takes mathematical expression in the C++ syntax
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

file = XDMFFile("test_periodic.xdmf")

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool ((near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and 
            (not ((near(x[0], a) and near(x[2], a)) or 
                  (near(x[0], a) and near(x[1], a)) or
                  (near(x[1], a) and near(x[2], a)))) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
    	#### define mapping for a single point in the box, such that 3 mappings are required
        if near(x[0], a) and near(x[1], a) and near(x[2],a):
            y[0] = x[0] - a
            y[1] = x[1] - a
            y[2] = x[2] - a
        ##### define mapping for edges in the box, such that mapping in 2 Cartesian coordinates are required
        if near(x[0], a) and near(x[2], a):
            y[0] = x[0] - a
            y[1] = x[1] 
            y[2] = x[2] - a      
        elif near(x[1], a) and near(x[2], a):
            y[0] = x[0] 
            y[1] = x[1] - a
            y[2] = x[2] - a
        elif near(x[0], a) and near(x[1], a):
            y[0] = x[0] - a
            y[1] = x[1] - a
            y[2] = x[2]         
        #### right maps to left: left/right is defined as the x-direction
        elif near(x[0], a):
            y[0] = x[0] - a
            y[1] = x[1]
            y[2] = x[2]
        ### back maps to front: front/back is defined as the y-direction    
        elif near(x[1], a):
            y[0] = x[0]
            y[1] = x[1] - a
            y[2] = x[2] 
        #### top maps to bottom: top/bottom is defined as the z-direction        
        elif near(x[2], a):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - a

## Define boundary condition
pbc = PeriodicBoundary()

V = FunctionSpace(mesh, 'Lagrange', 3, constrained_domain=pbc)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
f = Constant(-6.0)
L = f*v*dx

## Compute solution
u = Function (V)
solve(a == L, u, [])

file.write(u)