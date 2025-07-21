# import libraries
import numpy as np
import pandas as pd
import os
from scipy.special import expit  # logistic sigmoid

# iet random seed for reproducibility
random_state = 0
np.random.seed(random_state)

''' DEFINE CLEANING METHODS AND THEIR PARAMETER MEANS AND STANDARD DEVIATIONS '''
# these values are loosely based on typical industry reports for each cleaning method, but mostly arbitrary for synthetic data generation
methods = {
    "SC-1": {
        "roughness": (1.0, 0.2),  # surface roughness (nm RMS)
        "defects": (5e5, 1e5),    # defect density (#/cm²)
        "etch_rate": (10, 3),    # etch rate (nm/min)
        "Dit": (5e11, 1e11),     # interface trap density (cm⁻²·eV⁻¹)
        "Qf": (5e-6, 1e-6),      # fixed oxide charge (C/cm²)
        "lifetime": (20, 5),     # minority carrier lifetime (μs)
        "leakage": (1e-8, 5e-9), # leakage current (A/cm²)
        "breakdown": (9, 1),     # breakdown field (MV/cm)
        "pit_density": (500, 200)  # pit density (pits/cm²)
    },
    "RCA-1": {
        "roughness": (0.8, 0.15),
        "defects": (3e5, 8e4),
        "etch_rate": (8, 2),
        "Dit": (3e11, 8e10),
        "Qf": (4e-6, 8e-7),
        "lifetime": (30, 8),
        "leakage": (5e-9, 2e-9),
        "breakdown": (10, 0.8),
        "pit_density": (300, 150)
    },
    "UV/Ozone": {
        "roughness": (0.3, 0.1),
        "defects": (1e5, 5e4),
        "etch_rate": (0.5, 0.2),
        "Dit": (1e11, 5e10),
        "Qf": (2e-6, 5e-7),
        "lifetime": (50, 10),
        "leakage": (1e-10, 5e-11),
        "breakdown": (12, 0.5),
        "pit_density": (50, 30)
    },
    "HF Dip": {
        "roughness": (0.5, 0.12),
        "defects": (2e5, 6e4),
        "etch_rate": (2, 0.5),
        "Dit": (2e11, 6e10),
        "Qf": (3e-6, 6e-7),
        "lifetime": (40, 7),
        "leakage": (1e-9, 4e-10),
        "breakdown": (11, 0.7),
        "pit_density": (150, 80)
    },
    "Piranha": {
        "roughness": (1.2, 0.25),
        "defects": (7e5, 1.2e5),
        "etch_rate": (12, 4),
        "Dit": (6e11, 1.2e11),
        "Qf": (6e-6, 1.2e-6),
        "lifetime": (15, 4),
        "leakage": (2e-8, 6e-9),
        "breakdown": (8, 1.2),
        "pit_density": (600, 250)
    }
}

''' GENERATE SYNTHETIC DATA FOR WAFER SAMPLES '''
# number of synthetic wafer samples to generate
n = 50000

# prepare an empty list to store each wafer's synthetic data
data = []

# list of all available cleaning methods
methods_list = list(methods.keys())

# generate synthetic data for each sample
for _ in range(n - 1):
    # randomly choose a cleaning method for this sample
    method = np.random.choice(methods_list)
    # get the parameter distribution (mean, std) for the selected method
    params = methods[method]
    # create a synthetic row by sampling each metric from a normal distribution
    row = {
        'Cleaning_Method': method,
        'Surface_Roughness_nm': np.abs(np.random.normal(*params['roughness'])),
        'Defect_Density_per_cm2': np.abs(np.random.normal(*params['defects'])),
        'Etch_Rate_nm_per_min': np.abs(np.random.normal(*params['etch_rate'])),
        'Interface_Trap_Density_cm2_eV': np.abs(np.random.normal(*params['Dit'])),
        'Fixed_Oxide_Charge_C_per_cm2': np.abs(np.random.normal(*params['Qf'])),
        'Minority_Carrier_Lifetime_us': np.abs(np.random.normal(*params['lifetime'])),
        'PN_Junction_Leakage_A_per_cm2': np.abs(np.random.normal(*params['leakage'])),
        'Gate_Oxide_Breakdown_MV_per_cm': np.abs(np.random.normal(*params['breakdown'])),
        'Pit_Density_per_cm2': np.abs(np.random.normal(*params['pit_density']))
    }
    # append this synthetic row to the dataset
    data.append(row)

# convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

''' SIMULATE DAMAGE PROBABILITY USING A LOGISTIC FUNCTION BASED ON FEATURES '''

# assign weights to each feature (higher weight = more influence on damage)
weights = np.array([0.3, 0.2, 0.1, 0.15, 0.1, -0.05, 0.1, 0.1, 0.2])

# select features that influence damage prediction
features = df[['Surface_Roughness_nm', 'Defect_Density_per_cm2', 'Etch_Rate_nm_per_min',
               'Interface_Trap_Density_cm2_eV', 'Fixed_Oxide_Charge_C_per_cm2',
               'Minority_Carrier_Lifetime_us', 'PN_Junction_Leakage_A_per_cm2',
               'Gate_Oxide_Breakdown_MV_per_cm', 'Pit_Density_per_cm2']]

# normalize features (mean=0, std=1) to make weights comparable
feat_norm = (features - features.mean()) / features.std()

# apply linear combination of normalized features with weights
# subtract a bias term (-0.5) to shift the sigmoid curve
linear_combination = feat_norm.values.dot(weights) - 0.5

# convert the linear output into a probability (0 to 1) using the logistic sigmoid
df['Damage_Prob'] = expit(linear_combination)

# save the dataset as a CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')
df.to_csv(csv_path, index=False)

# print the first 5 rows of the DataFrame rounded to 3 decimals
print("Raw synthetic data generated and saved with random_state={random_state}")


