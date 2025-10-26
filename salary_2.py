import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import numpy as np

# Load your data
df = pd.read_csv("model_data.csv")

# Specify categorical columns
categorical_cols = [
    "EMST_TOGA", "RACETHM", "EMSECSM", "DGRDG", 
    "NDGMENG", "N2OCPRMG", "Years_To_First_Job",
    "EMTP", "SEX_2023", "FACIND", "FACSEC", "FACSOC"
]

# Separate features and target
X = df[["AGE"] + categorical_cols].copy()
y = df["SALARY"].astype(int)
feature_order = X.columns.tolist()
encoders = {}
"""
for col in categorical_cols:
    X[col] = X[col].astype("category")
    encoders[col] = list(X[col].cat.categories)  # save category order
    X[col] = X[col].cat.codes
"""
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1211
)

# Define exact R factor levels for each categorical column
r_levels = {
    "EMST_TOGA": ["85",  "87",  "88",  "90",  "91",  "92",
                    "93",  "96",  "97",  "98",
                        "166", "185", "249","374", "399"],        # replace with actual R levels
    "RACETHM": ["1","2","3","4","5","6","7"],
    "EMSECSM": ["1","2","3"],
    "DGRDG": ["1","2","3","4"],
    "NDGMENG": ["11", "12", "21", "22", "23",
                 "31", "32", "33", "34", "41",
                   "42", "43", "44", "45", "51", "52",
                 "53", "54", "55","56", "57", "61", "62", "63", "64",
                   "71", "72", "73", "74", "75", "76"],
    "N2OCPRMG": ["1","2","3","4","5","6","7"],
    "Years_To_First_Job": ["0","1","2","3","4","5"],
    "EMTP": ["01", "02", "03", "04", "05", "06", "10",
              "11", "12", "13", "14", "15", "16", "17", "18"],                       # example
    "SEX_2023": ["M","F"], 
    "FACIND": ["1","2","3","4"], 
    "FACSEC": ["1","2","3","4"], 
    "FACSOC": ["1","2","3","4"]
}




# Train model
model = RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=1211)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("RMSE:", np.sqrt(mse))

# Save model + encoders
joblib.dump({"model": model, "encoders": encoders, "feature_order": feature_order}, "salary_model_r_match.pkl")
