import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import joblib

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_excel("aimlproject.xlsx")
df.columns = df.columns.str.strip()

# ----------------------------
# CREATE FINANCIAL FEATURES
# ----------------------------
df['Income'] = df['Investment Portfolio Value'] / 10
df['Expenses'] = df['Average Monthly Spend on Entertainment'] * 2
df['Savings'] = df['Income'] - df['Expenses']

df['EMI'] = df['Expenses'] * 0.3
df['Loan'] = df['EMI'] * 12
df['Credit_Utilization'] = df['Risk Tolerance in Investments'] * 10

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df['savings_ratio'] = (df['Income'] - df['Expenses']) / df['Income']
df['expense_ratio'] = df['Expenses'] / df['Income']
df['emi_burden'] = df['EMI'] / df['Income']

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# ----------------------------
# TARGET VARIABLE
# ----------------------------
df['Stress_Level'] = pd.cut(
    df['Financial Wellness Index'],
    bins=[0, 40, 70, 100],
    labels=['High', 'Medium', 'Low'],
    include_lowest=True
)

df = df.dropna(subset=['Stress_Level'])

le = LabelEncoder()
df['Stress_Level'] = le.fit_transform(df['Stress_Level'])

# ----------------------------
# FEATURES
# ----------------------------
features = [
    'Income', 'Expenses', 'Savings',
    'Loan', 'EMI', 'Credit_Utilization',
    'savings_ratio', 'expense_ratio', 'emi_burden'
]

X = df[features]
y = df['Stress_Level']

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# CLASS BALANCING
# ----------------------------
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# ----------------------------
# MODEL
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight=class_weights,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

print("\n🔥 FINAL MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# SAVE EVERYTHING
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("✅ Model & test data saved!")