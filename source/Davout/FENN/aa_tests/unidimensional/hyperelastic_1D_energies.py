# Routine to store 1D hyperelastic energy density functions and their 
# derivatives

# Defines a function to get the derivatives of the Neo-Hookean energy 
# density in terms and with respect to the right Cauchy-Green strain 
# tensor

def neo_hookean(C, E):

    mu = 0.5*E

    # Evaluates the first derivative

    dPsi = 0.5*mu*(1-(1/C))

    # Evaluates the second derivative

    ddPsi = 0.5*(mu/(C**2))

    return dPsi, ddPsi