from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
import os

''' Parse Synthetic Data CSV File '''
# import as pandas dataframe
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "synthetic_data", "synthetic_data.csv")
df = pd.read_csv(csv_path)


# remove the Cleaning_Method column, redundant for analysis
df.drop('Cleaning_Method', axis=1, inplace=True)  


''' Traning Random Forest Regressor '''

# split the data into features (X) and target (y)
X = df.drop(['Damage_Prob'], axis=1)  # features
y = df['Damage_Prob']  # target variable

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# build the random forest regression model
rfc = RandomForestRegressor(n_estimators=30, random_state=0)

# fit the model to the training data
rfc.fit(X_train, y_train)

''' Model Testing and Evaluation '''

# make predictions on the test set
y_pred = rfc.predict(X_test)

# assess the model's performance using loss functions: mean squared error, mean absolute error, and R-squared, for both training and testing sets
y_train_pred = rfc.predict(X_train)
y_test_pred = rfc.predict(X_test)

metrics = {
    'Dataset': ['Training', 'Test'],
    'MSE': [
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)
    ],
    'MAE': [
        mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_test, y_test_pred)
    ],
    'R2': [
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)
    ]
}
# print metrics
print("Random Forest Regressor Performance Metrics:")
print(pd.DataFrame(metrics).T)

# write X_train, y_train, X_test, y_test, and y_pred to a csv file
test_results = X_test.copy()
test_results['y_test'] = y_test.values
test_results['y_pred'] = y_pred

out_path = os.path.join(script_dir, "..", "graphical_analysis", "results.csv")
test_results.to_csv(out_path, index=False)


