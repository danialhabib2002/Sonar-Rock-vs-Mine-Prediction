import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
 
# Read the data
data = pd.read_csv('Sonar data.csv',header=None)

# Print the first 5 rows of the data
print(data.head())

#number of rows and columns
print(data.shape)

#describe the data
print(data.describe())

# Split the data into  X  and y
X = data.drop(columns=60,axis=1)  # All rows, all columns except the last
y = data[60]   # All rows, only the last column


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create the model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict the labels of the test set
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print('Accuracy:', accuracy)


input = (0.0599,0.0474,0.0498,0.0387,0.1026,0.0773,0.0853,0.0447,0.1094,0.0351,0.1582,0.2023,0.2268,0.2829,0.3819,0.4665,0.6687,0.8647,0.9361,0.9367,0.9144,0.9162,0.9311,0.8604,0.7327,0.5763,0.4162,0.4113,0.4146,0.3149,0.2936,0.3169,0.3149,0.4132,0.3994,0.4195,0.4532,0.4419,0.4737,0.3431,0.3194,0.3370,0.2493,0.2650,0.1748,0.0932,0.0530,0.0081,0.0342,0.0137,0.0028,0.0013,0.0005,0.0227,0.0209,0.0081,0.0117,0.0114,0.0112,0.0100)
input_numpy = np.asarray(input)

# Reshape the array
input_reshape = input_numpy.reshape(1,-1)

# Standardize the input
std_input = scaler.transform(input_reshape)

# Predict the label of the standardized input
prediction = model.predict(std_input)

print(prediction)

if (prediction == 'R'):
  print('The object is a Rock')

else:
  print('The object is a Mine')