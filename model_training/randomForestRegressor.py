import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

''' Parse Synthetic Data CSV File '''

# import as pandas dataframe
df = pd.read_csv("C:/Users/aarav/OneDrive/Desktop/myCode/science_extension/synthetic_data/synthetic_data.csv")

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

# print model score
print(rfc.score(X_test, y_test))