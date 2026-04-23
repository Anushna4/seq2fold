import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from feature_extraction import extract_features

# Load dataset
df = pd.read_csv("../data/fold_data.csv")

# 🔥 Reduce to top 5 folds
top_folds = df['fold'].value_counts().nlargest(5).index
df = df[df['fold'].isin(top_folds)]

print("Using folds:", list(top_folds))
print("Dataset size:", df.shape)

# Extract features
X = np.array(df["sequence"].apply(extract_features).tolist())
y = df["fold"]

# 🔥 STEP 1: Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔥 STEP 2: PCA
pca = PCA(n_components=150)
X = pca.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM
model = SVC(kernel='rbf', C=30, gamma='auto', probability=True)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save models
joblib.dump(model, "../models/svm_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(pca, "../models/pca.pkl")

print("✅ Model, scaler, and PCA saved!")