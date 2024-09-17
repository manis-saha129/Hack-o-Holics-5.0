# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset)
# printing the first 5 rows of the dataset
print(diabetes_dataset.head())
# number of rows and columns in the dataset
print(diabetes_dataset.shape)
# getting the statistical measures of the data
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
# 0 --> Non-Diabetic
# 1 --> Diabetic
print(diabetes_dataset.groupby('Outcome').mean())
# separating the data and levels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

# Data Standardization
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

# Train, Test, Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Training the Model
classifier = svm.SVC(kernel='linear')
# training the Support Vector Machine Classifier
classifier.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score of the Training Data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score of the Test Data : ', test_data_accuracy)

# Making a Predictive System
input_data = (6,148,72,35,0,33.6,0.627,50)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scalar.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The Person is not Diabetic.')
else:
    print('The Person is Diabetic.')
