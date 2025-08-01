# import necessary libraries
import pandas as pd
import seaborn as sns
import os   
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

''' Parse results CSV File '''
# import as pandas dataframe
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the directory of the current script
csv_path = os.path.join(script_dir, "results.csv") # construct the path to the synthetic data CSV file
df = pd.read_csv(csv_path) # read the CSV file into a pandas DataFrame

# save y_test and y_pred to a pandas DataFrame for further analysis
y_test = df['y_test']
y_pred = df['y_pred']


''' Parse Feature Importances CSV File '''
feat_path = os.path.join(script_dir, "feature_importances.csv")
importances = np.loadtxt(feat_path, delimiter=",")

''' Predicted vs Actual Plot '''
plt.figure(figsize=(6, 4)) 
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs. Actual")
plt.savefig(os.path.join(script_dir, "predicted_vs_actual.png"), dpi=300, bbox_inches='tight')

''' Residuals Plot '''
plt.figure(figsize=(6, 4)) 
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted")
plt.savefig(os.path.join(script_dir, "residuals_plot"), dpi=300, bbox_inches='tight')

''' Residuals Histogram '''
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig(os.path.join(script_dir, "residuals_histogram"), dpi=300, bbox_inches='tight')

''' Q-Q Plot (Quantil_Quantile)'''
plt.figure(figsize=(6, 4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.savefig(os.path.join(script_dir, "qq_plot"), dpi=300, bbox_inches='tight')

''' Feature Importance Plot '''
plt.figure(figsize=(6, 4))
feat_names = df.columns
feat_names = feat_names[:-2]  # exclude 'y_test' and 'y_pred'
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importances")
plt.savefig(os.path.join(script_dir, "feature_importances"), dpi=300, bbox_inches='tight')
