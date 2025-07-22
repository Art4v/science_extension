# import libraries
import numpy as np
import pandas as pd
import os

# Load the original synthetic dataset as pandas dataframe
csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')
df = pd.read_csv(csv_path)

# Set random seed and save state
random_state = 0
rng = np.random.RandomState(random_state)
# Save the random state for reproducibility
tuple_state = rng.get_state()

# Define noise level as a fraction of each column's std deviation
noise_fraction = 0.40  # 40% noise

# List of numeric feature columns to perturb
target_cols = [
    'Surface_Roughness_nm', 'Defect_Density_per_cm2', 'Etch_Rate_nm_per_min',
    'Interface_Trap_Density_cm2_eV', 'Fixed_Oxide_Charge_C_per_cm2',
    'Minority_Carrier_Lifetime_us', 'PN_Junction_Leakage_A_per_cm2',
    'Gate_Oxide_Breakdown_MV_per_cm', 'Pit_Density_per_cm2',
]

# Add Gaussian noise to each column
for col in target_cols:
    std_dev = df[col].std()
    noise = rng.normal(loc=0, scale=noise_fraction * std_dev, size=df.shape[0])
    df[col] += noise


# Restore RNG state if needed elsewhere
rng.set_state(tuple_state)


# Save the noisy dataset 
csv_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')
df.to_csv(csv_path, index=False)

print(f"Dataset noised with random_state={random_state}")

