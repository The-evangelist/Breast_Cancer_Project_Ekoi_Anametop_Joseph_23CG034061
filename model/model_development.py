import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# 1. Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target  # 0 = malignant, 1 = benign

# 2. Feature Selection (Pick Any 5)
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean compactness'
]

X = df[selected_features]
y = df['diagnosis']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling (Important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model Training
model = SVC(kernel='rbf', probability=False)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 7. Save Model using Pickle
with open("model/breast_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Scaler using Pickle
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 8. Reload to Verify
with open("model/breast_cancer_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

# Test reloaded model
sample = X.iloc[0].values.reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)
print("Reloaded Prediction:", loaded_model.predict(sample_scaled))
