# import relevant libraries
import pandas as pd # for dataframes
import seaborn as sns # for visualisation of data normally, this is now used to grab a dataset 

# CREATE RAMDOM MODEL

healthexp = sns.load_dataset('healthexp') # import dataset for life expectancy

healthexp = pd.get_dummies(healthexp) # convert data to pandas dataframe for ease of use

# seperate data into independent (features) and dependent (target) variables for preprocessing before model training
X = healthexp.drop(['Life_Expectancy'], axis=1) # creates a dataframe X with all the data from healthexp, except for the column Life_Expectancy
y = healthexp['Life_Expectancy'] # creates a pandas series Y with only Life_Expectancy

# import train_test_split from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=19) # randomly distribute training and testing data into an 80/20 split

# import RandomForestRegressor from sklearn
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=13)  # builds the random forest regression model
rfr.fit(X_train, y_train) # fit training data to the model

y_pred = rfr.predict(X_test) # generate predictions based on text data, store in y_pred

# import metrics to analyse accuracy of our model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# print metrics
print('Mean Absolulte Error RAW: ', mean_absolute_error(y_pred, y_test))
print('Mean Squared Error RAW: ', mean_squared_error(y_pred, y_test)) 
print('R2 Score RAW: ', r2_score(y_pred, y_test))

# NOW ADDING MOFIEIED PARAMETERS

# define a parameter grid for the parametors
param_grid = {
    'n_estimators': [100, 200, 300],  # number of decision trees
    'max_depth': [10, 20, 30], # depth of each tree
    'min_samples_split': [2, 5, 10], # minimal number of samples to split an internal node
    'min_samples_leaf': [1, 2, 4] # minimal number of samples to be at the leaf node
}

# import GridSearchCV -- a hyperparameter tuning tool 
from sklearn.model_selection import GridSearchCV

# create trained model
rfr_cv = GridSearchCV(
    estimator = rfr, # use this random forest regressor as the model that should be tuned with different hyperparameter
    param_grid=param_grid, # use the paramgrid defined earlier 
    cv = 3, # use three fold cross validation
    scoring="neg_mean_squared_error", # use neg_mean_squared_error to evaluate the performance of the model
    n_jobs=-1 # use all available cores on the machine to train
)

rfr_cv.fit(X_train, y_train) # fit training data to model

y_pred = rfr_cv.predict(X_test) # generate predictions based on text data, store in y_pred

# output new parametrics
print()
print('Mean Absolulte Error Trained: ', mean_absolute_error(y_pred, y_test))
print('Mean Squared Error Trained: ', mean_squared_error(y_pred, y_test)) 
print('R2 Score Trained: ', r2_score(y_pred, y_test))