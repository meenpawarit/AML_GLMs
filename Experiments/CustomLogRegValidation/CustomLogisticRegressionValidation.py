# Run this script from the root directory of the project, python3 Experiments/CustomLogRegValidation/CustomLogisticRegressionValidation.py

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from CustomLogisticRegression import CustomLogisticRegression

data = pd.read_csv("Experiments/Data/processed_cleveland_python.csv")
data["target"] = np.where(data["target"] == 0, 0, 1)

# Categorical variables
categorical_cols = ['cp', 'sex','fbs', 'restecg', 'exang', 'slope', 'thal']

y = data["target"].values
X = data.drop(columns=["target"])

# Build feature groups
feature_groups = {}
for col in categorical_cols:
    dummy_cols = [c for c in X.columns if c.startswith(col + "_")]
    feature_groups[col] = dummy_cols

numeric_cols = [c for c in X.columns if not any(c.startswith(cat + "_") for cat in categorical_cols)]
for col in numeric_cols:
    feature_groups[col] = [col]
X_np = X.values
feature_names = list(X.columns)
full_model = CustomLogisticRegression(max_iter=100, tol=1e-6, verbose=False)
full_model.fit(X_np, y, feature_names=feature_names, feature_groups=feature_groups)
full_model.summary()
step_model = full_model.step(direction="backward", criterion="AIC")
step_model.summary()
print(step_model.get_coefficients())
