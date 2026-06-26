from fastapi import FastAPI,HTTPException
from pydantic import BaseModal
import joblib
print("library impoerted ")
import numpy as np 
import pandas as pd 
app=FastAPI(title='financity ai recomendation',version="1.0")
try:
    model = joblib.load('models/rf_category_predictor.pkl')
    feature_encoders = joblib.load('models/feature_encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    print("🚀 AI Models & Translators loaded successfully into memory!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {str(e)}")
class UserInput(BaseModel):
    Age: int
    Gender: str
    Qualification: str
    Financial_Goal: str
@app.get("/")
def home():
    return {"status": "Online", "message": "Financity Recommendation Microservice is running smoothly."}
@app.post("/predict")
def predict_financial_path(user_data: UserInput):
    try:
        input_dict = {
            'Age': [user_data.Age],
            'Gender': [user_data.Gender],
            'Qualification': [user_data.Qualification],
            'Financial_Goal': [user_data.Financial_Goal]
        }
        input_df = pd.DataFrame(input_dict)
        for col in ['Gender', 'Qualification', 'Financial_Goal']:
            encoder = feature_encoders[col]
            #unknown value handling
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid categorical value provided for {col}")
        predicted_code = model.predict(input_df)[0]
        final_recommended_path = target_encoder.inverse_transform([predicted_code])[0]
        return {
            "status": "success",
            "demographics": {
                "age": user_data.Age,
                "qualification": user_data.Qualification,
                "goal": user_data.Financial_Goal
            },
            "recommended_path": final_recommended_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

