ML
                      ####################  Slip 1/25 #######################

Q.1 Write a Python program to transform data with Principal Component Analysis (PCA). 
Use a handwritten digit dataset. 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
digits = load_digits()
X = digits.data  # Features
y = digits.target  # Labels

# Standardize the data to have mean = 0 and variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions (reduce to 2 for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
    
# Plot the 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter, label='Digit Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Handwritten Digits Dataset')
plt.show()

                 #####################  Slip 2/24 ##############################

Q.1 Write a Python program to implement simple Linear Regression for predicting house 
price.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset: House size (in 1000 sq ft) and corresponding price (in $100,000)
# X represents house sizes, y represents house prices
X = np.array([[1.1], [1.3], [1.5], [2.0], [2.2], [2.9], [3.0], [3.2], [3.6], [4.0], [4.5]])
y = np.array([150, 180, 220, 250, 280, 330, 360, 400, 420, 500, 550])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for the test set
y_pred = model.predict(X_test)

# Visualize the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('House Size (1000 sq ft)')
plt.ylabel('House Price ($100,000)')
plt.title('Simple Linear Regression: House Price Prediction')
plt.legend()
plt.show()

# Print model coefficients
print(f"Slope (Coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Example: Predict price for a new house with size 2.5 (1000 sq ft)
new_house_size = np.array([[2.5]])
predicted_price = model.predict(new_house_size)
print(f"Predicted price for house of size 2.5 (1000 sq ft): ${predicted_price[0] * 1000:.2f}")




                     ########################  Slip 3/23 ###########################

Q.1 Write a Python program to implement multiple Linear Regression for predicting
 house price. 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'Age': [20, 15, 10, 5, 25, 20, 10, 5, 30, 15],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features (Size, Bedrooms, Age) and target variable (Price)
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print model performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict price for a new house with example features
new_house = np.array([[1500, 3, 15]])  # Size=1500 sq ft, Bedrooms=3, Age=15 years
predicted_price = model.predict(new_house)
print(f"Predicted price for a new house: ${predicted_price[0]:.2f}")



                   ####################### Slip 4/22 ##############################

Q.1 Write a Python program to implement logistic Regression for predicting
 whether a person will buy the insurance or not. Use insurance_data.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('insurance_data.csv')

# Features (e.g., Age and Income) and Target variable (Buy_Insurance: 1 if yes, 0 if no)
X = data[['Age', 'Income']]
y = data['Buy_Insurance']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example: Predict if a person with Age=30 and Income=50000 will buy insurance
new_person = [[30, 50000]]
prediction = model.predict(new_person)
print("Will the person buy insurance?", "Yes" if prediction[0] == 1 else "No")


                       ###################   Slip 5/21 #########################

Q.1 Write a Python program to implement logistic Regression for a handwritten digit dataset.
 
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the handwritten digits dataset
digits = load_digits()
X = digits.data  # Features (image pixels)
y = digits.target  # Target (digit labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example prediction for a single image
import matplotlib.pyplot as plt

# Display an example image from the test set and predict its label
plt.imshow(X_test[0].reshape(8, 8), cmap='gray')
plt.title("Example Image")
plt.show()
example_prediction = model.predict([X_test[0]])
print(f"Predicted label for the example image: {example_prediction[0]}")


                        #######################  slip 6/20 #########################

Q.1 Write a Python program to implement Polynomial Regression for positionsal.csv 
dataset.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('positionsal.csv')

# Assuming columns 'Level' and 'Salary' in positionsal.csv
X = data[['Level']].values
y = data['Salary'].values

# Polynomial transformation (degree 4 is chosen here; you can adjust it as needed)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual Salary')  # Original data points
plt.plot(X, model.predict(X_poly), color='red', label='Polynomial Regression Fit')  # Polynomial fit line
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Polynomial Regression for Position Level vs. Salary')
plt.legend()
plt.show()

# Example: Predict the salary for a specific position level (e.g., level 5.5)
example_level = [[5.5]]
example_level_poly = poly.transform(example_level)
predicted_salary = model.predict(example_level_poly)
print(f"Predicted salary for level 5.5: ${predicted_salary[0]:.2f}")




                       ################## Slip 7/19 ####################3

Q.1 Write a Python program to implement Decision Tree Model for classification. 
Use Decision_Tree_Dataset.csv 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Decision_Tree_Dataset.csv')

# Assume the dataset has feature columns labeled as 'feature1', 'feature2', ..., and a 'target' column
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



                       ################### Slip 8/18 #############################3


Q.1 Write a Python program to implement linear SVM for Regression. Use 
position_sal.csv. 

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('position_sal.csv')

# Assume columns 'Level' and 'Salary' are present in the dataset
X = data[['Level']].values  # Features
y = data['Salary'].values  # Target variable

# Scaling features for better performance in SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Apply Linear SVR
svr = SVR(kernel='linear')
svr.fit(X_scaled, y_scaled)

# Predict and reverse scaling for interpretation
y_pred_scaled = svr.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, y_pred, color='red', label='Linear SVR Prediction')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Linear SVM Regression for Position Level vs. Salary')
plt.legend()
plt.show()

# Example: Predict salary for a specific level (e.g., level 5.5)
example_level = np.array([[5.5]])
example_level_scaled = scaler_X.transform(example_level)
predicted_salary_scaled = svr.predict(example_level_scaled)
predicted_salary = scaler_y.inverse_transform(predicted_salary_scaled)
print(f"Predicted salary for level 5.5: ${predicted_salary[0]:.2f}")



                          ######################  Slip 9/17 ##########################
Q.1 Write a Python program to implement linear SVM for Classification. Use 
iris.csv. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('iris.csv')

# Assume the dataset has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
y = data['species']  # Target labels

# Encode the species labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create and train the Linear SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



                           ####################### Slip 10/16 ################################

Q.1 Write a Python program to implement k-nearest Neighbors algorithm to build a
prediction model. Use Iris Dataset. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('iris.csv')

# Assume the dataset has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
y = data['species']  # Target labels

# Encode the species labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Create and train the KNN classifier (using 5 neighbors)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



                               ################### Slip 11/15  ###################

Q.1 Write a Python program to implement Naïve Bayes for classification. Use 
titanic.csv/spam.csv dataset. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Titanic dataset
data = pd.read_csv('titanic.csv')

# Display first few rows of the dataset
print(data.head())

# Preprocess the data: Drop columns that are not needed and handle missing values
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
data = data.dropna()

# Encode categorical features (e.g., 'Sex')
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Define the features (X) and target (y)
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']]
y = data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Gaussian Naive Bayes model
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



                                 #################  Slip 12  #####################


Q.1 Write a Python program to implement k-means algorithm. Use income.csv 
dataset. 

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('income.csv')

# Display the first few rows to understand the structure of the data
print(data.head())

# Assuming the dataset contains a column 'Income' for clustering
income_data = data[['Income']]  # Selecting only the 'Income' column

# Perform KMeans clustering (assuming we want 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(income_data)

# Display the clusters
print(data.head())

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Income'], [0]*len(data), c=data['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.title('K-Means Clustering of Income Data')
plt.show()


                    ################### slip 13 #########################


Q.1 Write a Python program to implement Agglomerative clustering on a income.csv 
dataset. 

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('income.csv')

# Display the first few rows to understand the structure of the data
print(data.head())

# Assuming the dataset contains a column 'Income' for clustering
income_data = data[['Income']]  # Selecting only the 'Income' column

# Perform Agglomerative Clustering (assuming we want 3 clusters)
agg_clustering = AgglomerativeClustering(n_clusters=3)
data['Cluster'] = agg_clustering.fit_predict(income_data)

# Display the clusters
print(data.head())

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Income'], [0]*len(data), c=data['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.title('Agglomerative Clustering of Income Data')
plt.show()

 
                      ####################### Slip 14 ################


Q.1 Write a Python program to implement k-means algorithm on a synthetic dataset 
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-Means algorithm
kmeans = KMeans(n_clusters=4)
y_kmeans = kmeans.fit_predict(X)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.title('K-Means Clustering on Synthetic Dataset')
plt.show()



                       ##################  Slip 15 ###################


Q.1 Write a Python program to implement Naïve Bayes for classification. 
Use titanic.csv/spam.csv dataset. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Titanic dataset (ensure you have the titanic.csv file)
data = pd.read_csv('titanic.csv')

# Preprocessing
# For simplicity, let's drop rows with missing values and use relevant columns only
data = data.dropna(subset=['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])

# Convert categorical columns to numerical (e.g., Sex -> 0 for male, 1 for female)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and target variable
X = data[['Pclass', 'Sex', 'Age', 'Fare']]  # Features
y = data['Survived']  # Target variable (0 = Not Survived, 1 = Survived)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


                         #####################  Slip 16 #############################

Q.1 Write a Python program to implement k-nearest Neighbors algorithm to build a
prediction model. Use Iris Dataset. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=3 (you can change k for tuning)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


                         ############################# Slip 17 ########################

Q.1 Write a Python program to implement linear SVM for Classification. Use iris.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear SVM classifier
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


                       #########################   Slip 18 ############################


Q.1 Write a Python program to implement linear SVM for Regression. Use 
position_sal.csv 

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('position_sal.csv')

# Assume 'Position' is categorical and 'Level' and 'Salary' are numerical
X = data['Level'].values.reshape(-1, 1)  # Feature (Level)
y = data['Salary'].values  # Target (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear SVM for regression (SVR)
svr = SVR(kernel='linear')

# Train the model
svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, svr.predict(X), color='red', label='Regression line')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Linear SVM for Regression')
plt.legend()
plt.show()



                            ######################### Slip 19 ########################


Q.1 Write a Python program to implement Decision Tree Model for classification. 
Use Decision_Tree_Dataset.csv. 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('Decision_Tree_Dataset.csv')

# Assuming the last column is the target and others are features
X = data.iloc[:, :-1]  # Features (all columns except the last)
y = data.iloc[:, -1]   # Target (last column)

# Encode categorical variables (if any)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, plot the decision tree (requires graphviz and pydot)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=label_encoder.classes_)
plt.title("Decision Tree Classifier")
plt.show()



                           #####################  Slip 20 ######################


Q.1 Write a Python program to implement Polynomial Regression. Use 
position_sal.csv. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('position_sal.csv')

# Assume the first column is the level and the second column is the salary
X = data.iloc[:, 1:2].values  # Independent variable (position level)
y = data.iloc[:, 2].values    # Dependent variable (salary)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features (degree = 4 for this example)
poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)

# Create the model and fit it to the training data
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict the test set results
X_poly_test = poly.transform(X_test)
y_pred = model.predict(X_poly_test)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(poly.transform(X)), color='red')
plt.title('Polynomial Regression: Salary vs Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Output the results
print("Predicted salary values:", y_pred)




                   ##########################      Slip 21    ############################


Q.1 Write a Python program to implement logistic Regression for handwritten digit 
dataset. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the MNIST-like handwritten digit dataset (from sklearn)
digits = load_digits()

# Features and labels
X = digits.data  # Image pixel data
y = digits.target  # Target labels (digits 0-9)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display some test results
for i in range(5):
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]} | Actual: {y_test[i]}")
    plt.show()




                         ########################   Slip 22 ##############################


Q.1 Write a Python program to implement logistic Regression for predicting whether 
a person will buy the insurance or not. Use insurance_data.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the insurance data CSV (assumed file path)
data = pd.read_csv('insurance_data.csv')

# Explore the data (optional step to check the columns and types)
print(data.head())

# Assume the dataset has columns 'age', 'gender', 'income', and 'purchased' as target (0: No, 1: Yes)
# Preprocess the data (e.g., encoding categorical variables, handling missing values)
data['gender'] = data['gender'].map({'male': 0, 'female': 1})  # Encoding gender (0: male, 1: female)

# Features (input) and target (output)
X = data[['age', 'gender', 'income']]  # Input features
y = data['purchased']  # Target variable (whether insurance was purchased)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (scaling the features to mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Logistic Regression model: {accuracy * 100:.2f}%')




           /////////////////////// Slip 23 ////////////////////


Q.1 Write a Python program to implement multiple Linear Regression for predicting 
house price. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the house price dataset (assuming it's in a CSV file 'house_price_data.csv')
data = pd.read_csv('house_price_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Assume the dataset has the following columns: 'rooms', 'sqft_living', 'sqft_lot', 'location', and 'price'
# 'price' is the target variable

# Preprocessing the data (converting categorical variables if necessary)
# Example: Encoding a categorical column 'location' using one-hot encoding
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Features (input) and target (output)
X = data.drop('price', axis=1)  # Features (drop the target column)
y = data['price']  # Target variable (house price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (scaling the features to mean=0, variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of Multiple Linear Regression model: {mse:.2f}')


                 ////////////////////// Slip 24 ////////////////////////
 
Q.1 Write a Python program to implement simple Linear Regression for predicting 
house price. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the house price dataset (assume it's in a CSV file 'house_price_data.csv')
data = pd.read_csv('house_price_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Assume the dataset has columns: 'rooms' (number of rooms) and 'price' (house price)
# We will use 'rooms' to predict 'price'

# Features (input) and target (output)
X = data[['rooms']]  # Feature (number of rooms)
y = data['price']    # Target variable (house price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Visualizing the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Simple Linear Regression: House Price Prediction')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price')
plt.legend()
plt.show()


           ////////////////// Slip 25 ////////////////////


Q.1 Write a Python program to transform data with Principal Component Analysis 
(PCA). Consider handwritten digit dataset. 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
digits = load_digits()
X = digits.data  # Features
y = digits.target  # Labels

# Standardize the data to have mean = 0 and variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions (reduce to 2 for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter, label='Digit Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Handwritten Digits Dataset')
plt.show()

