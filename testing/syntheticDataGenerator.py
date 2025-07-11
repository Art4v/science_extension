''' this is a personal training exercise on how to generate syntehtic data '''

import numpy as np
import pandas as pd

# define number of samples 
n_samples = 1000

# model data distributions
surface_roughness = np.random.normal(loc=1.327, scale=0.9, size=n_samples) # generates random variables for surface_roughness
current_flow = np.random.normal(loc=0.67, scale=0.13, size=n_samples) # generates random variables for current_flow
bonding_time = np.random.normal(loc=777.77, scale=189.252, size = n_samples) # generates random variables for bonding_time 

# apply constraints and post processing
surface_roughness = np.clip(surface_roughness, 0.1, 3.0) # clips surface_roughness
current_flow = np.clip(current_flow, 0.5, 1.2) # clips current_flow
bonding_time = np.clip(bonding_time, 500, 1100) # clips bonding_time


df = pd.DataFrame({
    "surface_roughness (nm)": surface_roughness,
    "current_flow (mA)": current_flow,
    "bonding_time (s)": bonding_time,
})

print(df)

