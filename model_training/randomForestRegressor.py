# import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import os
import numpy as np

''' Parse Synthetic Data CSV File '''
# import as pandas dataframe
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the directory of the current script
csv_path = os.path.join(script_dir, "..", "synthetic_data", "synthetic_data.csv") # construct the path to the synthetic data CSV file
df = pd.read_csv(csv_path) # read the CSV file into a pandas DataFrame


# remove the Cleaning_Method column, redundant for analysis
df.drop('Cleaning_Method', axis=1, inplace=True) 

''' Traning Random Forest Regressor '''

# split the data into features (X) and target (y). The target is the value we want to predict based on the features
X = df.drop(['Damage_Prob'], axis=1)  # features
y = df['Damage_Prob']  # target variable

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# build the random forest regression model
rfr = RandomForestRegressor(n_estimators=30, random_state=0) # n_estimators is the number of trees in the forest, random_state ensures reproducibility

# fit the model to the training data
rfr.fit(X_train, y_train)

''' Model Testing and Evaluation '''

# make predictions on the test set
y_pred = rfr.predict(X_test)

# assess the model's performance using loss functions: mean squared error, mean absolute error, and R-squared, for both training and testing sets
y_test_pred = rfr.predict(X_test) # make predictions on the test set

metrics = {
    'MSE': [
        mean_squared_error(y_test, y_test_pred) # mean squared error for the test set
    ],
    'MAE': [
        mean_absolute_error(y_test, y_test_pred) # mean absolute error for the test set
    ],
    'R2': [
        r2_score(y_test, y_test_pred) # R-squared for the test set
    ]
}
# print metrics
print("Random Forest Regressor Performance Metrics:")
print(pd.DataFrame(metrics).T)

# write X_train, y_train, X_test, y_test, y_pred, and feature_importances to a csv file
test_results = X_test.copy()
test_results['y_test'] = y_test.values
test_results['y_pred'] = y_pred

out_path = os.path.join(script_dir, "..", "graphical_analysis", "results.csv")
test_results.to_csv(out_path, index=False)

# transfer feature importances through csv file
importances = rfr.feature_importances_
feat_path = os.path.join(script_dir, "..", "graphical_analysis", "feature_importances.csv")
np.savetxt(feat_path, importances, delimiter=",")


