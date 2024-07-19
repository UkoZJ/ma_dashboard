# %%

import numpy as np


# Main functional forms
def Briere_func(cte, tmin, tmax, temp):
    outp = temp * cte * (temp - tmin) * (tmax - temp) ** 0.5
    return max(outp, 0) if not np.isnan(outp) else 0


def Quad_func(cte, tmin, tmax, temp):
    outp = -cte * (temp - tmin) * (temp - tmax)
    return max(outp, 0) if not np.isnan(outp) else 0


def QuadN_func(cte, c1, c2, temp):
    outp = cte * temp**2 + c1 * temp + c2
    return max(outp, 0) if not np.isnan(outp) else 0


# Function to compute the hatching rate
def h_f(hum, rain):
    erat = 0.5
    e0 = 1.5
    evar = 0.05
    eopt = 8.0
    efac = 0.01
    edens = 0.01

    hatch = (1 - erat) * (
        ((1 + e0) * np.exp(-evar * (rain - eopt) ** 2))
        / (np.exp(-evar * (rain - eopt) ** 2) + e0)
    ) + erat * (edens / (edens + np.exp(-efac * hum)))
    return hatch


# Parameters for Aedes albopictus and Aedes aegypti
params = {
    "albopictus": {
        "a_f": (Briere_func, 0.000193, 10.25, 38.32),
        "TFD_f": (Briere_func, 0.0488, 8.02, 35.65),
        "pLA_f": (Quad_func, 0.002663, 6.668, 38.92),
        "MDR_f": (Briere_func, 0.0000638, 8.6, 39.66),
        "lf_f": (Quad_func, 1.43, 13.41, 31.51),
        "dE_f": (Quad_func, 0.00071, 1.73, 40.51),
        "deltaE_f": (QuadN_func, 0.0019328, -0.091868, 1.3338874),
    },
    "aegypti": {
        "a_f": (Briere_func, 0.000202, 13.35, 40.08),
        "TFD_f": (Briere_func, 0.00856, 14.58, 34.61),
        "pLA_f": (Quad_func, 0.004186, 9.373, 40.26),
        "MDR_f": (Briere_func, 0.0000786, 11.36, 39.17),
        "lf_f": (Quad_func, 0.148, 9.16, 37.73),
        "dE_f": (Briere_func, 0.0003775, 14.88, 37.42),
        "deltaE_f": (QuadN_func, 0.004475, -0.210787, 2.55237),
    },
}


def compute_param(func, *args):
    return func(*args)


# Generic RM function
def R0_func(species, Te, rain, hum):
    if np.isnan(Te) or np.isnan(rain) or np.isnan(hum):
        return np.nan

    species_params = params[species]

    a = compute_param(*species_params["a_f"], Te)
    f = (1.0 / 2.0) * compute_param(*species_params["TFD_f"], Te)
    deltaa = compute_param(*species_params["lf_f"], Te)
    dE = compute_param(*species_params["dE_f"], Te)
    probla = compute_param(*species_params["pLA_f"], Te)
    h = h_f(hum, rain)
    deltaE = compute_param(*species_params["deltaE_f"], Te)

    R0 = ((f * a * deltaa) * probla * ((h * dE) / (h * dE + deltaE))) ** (1.0 / 3.0)
    return R0


import numpy as np

# Example usage of R0_func

# Define temperature, rainfall, and human density
temperature = 25  # degrees Celsius
rainfall = 10  # mm
human_density = 1000  # people per square km

# Compute R0 for Aedes albopictus
R0_albopictus = R0_func("albopictus", temperature, rainfall, human_density)
print(f"Suitability index for Aedes albopictus: {R0_albopictus}")

# Compute R0 for Aedes aegypti
R0_aegypti = R0_func("aegypti", temperature, rainfall, human_density)
print(f"Suitability index for Aedes aegypti: {R0_aegypti}")

# %%
