import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


#Load Dataset
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)

# Create directory for models
os.makedirs("models", exist_ok=True)

# Train and save Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_model.pkl") #Use joblib for large ML models

# Train and save Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest_model.pkl")

# Train and save KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, "models/knn_model.pkl")

# Print accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test)))