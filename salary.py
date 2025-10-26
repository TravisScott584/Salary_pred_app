import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib




df = pd.read_csv("c:/Users/Jackf/Analytics/R Files/AI101 Fin Proj/PYTHON/salary_train_data.csv")

print(df.head())
print(df.dtypes)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read CSV
df = pd.read_csv("c:/Users/Jackf/Analytics/R Files/AI101 Fin Proj/PYTHON/salary_train_data.csv")

# Force correct dtypes
df = df.astype({
    "AGE" : "int",
    "EMST_TOGA" : "string",
    "RACETHM" : "string",
    "EMSECSM" : "string",
    "DGRDG" : "string",
    "NDGMENG" : "string",
    "N2OCPRMG" : "string",
    "Years_To_First_Job" : "string",
    "EMTP" : "string",
    "SEX_2023" : "string",
    "FACIND" : "string",
    "FACSEC" : "string",
    "FACSOC" : "string",
    "SALARY" : "int"
})
categorical_cols = [
    "EMST_TOGA", "RACETHM", "EMSECSM", "DGRDG", 
    "NDGMENG", "N2OCPRMG", "Years_To_First_Job",
    "EMTP", "SEX_2023", "FACIND", "FACSEC", "FACSOC"
]

# Force categorical cols to string
df[categorical_cols] = df[categorical_cols].astype("category")
# Separate features and target
X = df[["AGE","EMST_TOGA","RACETHM","EMSECSM",
    "DGRDG","NDGMENG","N2OCPRMG","Years_To_First_Job",
    "EMTP","SEX_2023","FACIND","FACSEC","FACSOC"]].copy()   # 👈 make a deep copy

y = df["SALARY"]

# Encode categorical columns
encoders = {}
for col in X.select_dtypes(include=["category"]).columns:
    le = LabelEncoder() 
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le   # store encoder for later use

print(X.head())
print(X.dtypes)


model = RandomForestRegressor(
    n_estimators=200,   # number of trees
    max_depth=None,     # let it grow fully
    random_state=42
)

model.fit(X, y)

joblib.dump({"model": model, "encoders": encoders}, "salary_model2.pkl")
