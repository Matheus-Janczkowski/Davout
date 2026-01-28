# Routine to store optimization tools

import scipy.optimize as optimize

import numpy as np

# Defines a class to wrap scipy methods for unconstrained optimization
# using gradient information only

class UnconstrainedFirstOrderMinimization:

    def __init__(self, objective_function=None, objective_gradient=None,
    optimization_method="CG", initial_guess=None, n_design_variables=
    None, verbose=True):

        # Defines the objective function as a function of the arguments

        if objective_function is None:

            raise ValueError("The 'objective_function' input of the cl"
            "ass 'UnconstrainedFirstOrderMinimization' is not provided")
        
        else:

            self.objective_function = objective_function
        
        # Defines the gradient as a function of the arguments 

        if objective_gradient is None:

            raise ValueError("The 'objective_gradient' input of the cla"
            "ss 'UnconstrainedFirstOrderMinimization' is not provided")
        
        else:

            self.objective_gradient = objective_gradient

        # Sets the optimization method

        self.optimizer = optimization_method

        # Sets the initial guess as the design variables

        if initial_guess is None:

            if n_design_variables is None:

                raise ValueError("The 'initial_guess' is None, as is '"+
                "n_design_variables'. One or the other must be provide"+
                "d to 'UnconstrainedFirstOrderMinimization'")
            
            # Initializes the initial guess as a vector of zeros
            
            initial_guess = np.zeros(n_design_variables)

        self.design_variables = initial_guess

        # Sets the flag to show result information or not

        self.verbose = verbose

    # Defines a function to call the optimization process

    def __call__(self):
        
        # Calls the optimizer inside scipy

        result = optimize.minimize(self.objective_function, 
        self.design_variables, method=self.optimizer, jac=
        self.objective_gradient)

        # Shows information

        if self.verbose:

            if result.success:

                print("The optimization procedure was successful.\n"+
                str(result.message)+"\n")

            else:

                print("The optimziation procedure did not converge. Se"+
                "e:\n"+str(result.message)+"\n")

        # Updates the design variables

        self.design_variables = result.x