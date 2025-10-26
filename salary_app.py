import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

saved = joblib.load("salary_model2.pkl")
model = saved["model"]
encoders = saved["encoders"]

AGE = 21
EMST_TOGA = 90
RACETHM = 5
EMSECSM = 3
DGRDG = 1
NDGMENG = 75
N2OCPRMG = 7
Years_To_First_Job = 0
EMTP = 10
SEX_2023 = "M"
FACIND = 2
FACSEC = 4
FACSOC = 4

input_data = pd.DataFrame([{
    "AGE" : AGE,
    "EMST_TOGA" : EMST_TOGA,
    "RACETHM" : RACETHM,
    "EMSECSM" : EMSECSM,
    "DGRDG" : DGRDG,
    "NDGMENG" : NDGMENG,
    "N2OCPRMG" : N2OCPRMG,
    "Years_To_First_Job" : Years_To_First_Job,
    "EMTP" : EMTP,
    "SEX_2023" : SEX_2023,
    "FACIND" : FACIND,
    "FACSEC" : FACSEC,
    "FACSOC" : FACSOC
}])



for col, le in encoders.items():
    # Only add "MISSING" if not already in classes
    if "MISSING" not in le.classes_:
        le.classes_ = np.append(le.classes_, "MISSING")  # keep it as np.ndarray
    
    # Replace unseen values
    input_data[col] = input_data[col].apply(
        lambda x: x if x in le.classes_ else "MISSING"
    )
    
    # Transform
    input_data[col] = le.transform(input_data[col])



# 4. Predict
prediction = model.predict(input_data)[0]
print(f"💰 Predicted Salary: ${prediction:,.0f}")