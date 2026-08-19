# Routine to store methods for statistics

from scipy import stats

import numpy as np

# Defines a function to compute the confidence radius using Student's t
# distribution

def get_confidence_radius_student(n_samples, standard_deviation,
confidence_level=0.95):

    # Verifies the confidence level consistency

    if ((not isinstance(confidence_level, float)) and (not isinstance(
    confidence_level, int))) or confidence_level<=0 or (confidence_level
    )>1.0:

        raise ValueError("'confidence_level' in 'get_confidence_radius"+
        "_student' mus t be an integer or a float in the interval (0,1"+
        "]. Currently, it is: "+str(confidence_level))

    confidence_radius = (stats.t.ppf(0.5*(1.0+confidence_level), 
    n_samples-1)*(standard_deviation/np.sqrt(n_samples-2)))

    return confidence_radius