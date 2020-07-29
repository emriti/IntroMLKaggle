import numpy as np

# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = 'input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)

### PREDICTION MODEL
print("Making prediction for the following 5 houses: ")
print(X.head())
print("The prediction are ...")
print(melbourne_model.predict(X.head()))

### DEL.ME
test_data = [(2, 1.0, 156.0, -37.8079, 144.9934)]
test_data = np.array(test_data)

print("Making prediction for the following houses: ")
print(test_data)
print("The prediction are ...")
print(melbourne_model.predict(test_data))