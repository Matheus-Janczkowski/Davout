import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Data

t = np.array([1, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 10, 1000])

q = np.array([0.141, 0.196, 0.214, 0.229, 0.249, 0.263, 0.281, 0.298, 0.307, 0.312, 1/3])

# Exponential model

def model(t, a0, a1, a2, a3, a4):

    return 1/3 - np.exp(a0+(a1*t)+(a2*(t**2)))

#spline = UnivariateSpline(t, q, s=0)

# Set bounds: a must be negative, b and c are unbounded

bounds = ([-np.inf, -np.inf, -0.01, -np.inf, -np.inf], [np.inf, np.inf, 0.0, np.inf, np.inf])  # a <= 0

# Curve fitting

t_fit = np.linspace(min(t), max(t)*0.05, 300)

"""params, _ = curve_fit(model, t, q, bounds=bounds)

# Generate fit


#q_fit = spline(t_fit)

q_fit = model(t_fit, *params)"""

def model_interpolation(z):

    for i in range(len(t)):

        if t[i]>z:

            a1 = ((q[i]-q[i-1])/(t[i]-t[i-1]))

            a0 = q[i-1]-(a1*t[i-1])

            return a0+(a1*z)

q_fit = [model_interpolation(t_in) for t_in in t_fit]

# Plot

plt.plot(t, q, 'o', label='Data')

plt.plot(t_fit, q_fit, '-')

plt.xlabel('t')

plt.ylabel('q(t)')

#plt.title('a='+format(params[2],'.3e')+', b='+format(params[1],'.3e')+', c='+format(params[0],'.3e'))

plt.grid(True)

plt.legend()

plt.show()
