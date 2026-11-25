# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required Python libraries and load the Iris dataset.

2. Separate the input features and target species.

3. Split the dataset into training and testing data, then train the SGD Classifier model.

4. Predict the Iris flower species and calculate the model accuracy.

## Program:
```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier model
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Species:", y_test)
print("Predicted Species:", y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Iris Species:", iris.target_names[prediction[0]])
```

## Output:
<img width="840" height="110" alt="image" src="https://github.com/user-attachments/assets/823f55d7-5ef1-442e-8bf7-5259b112ebc2" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
