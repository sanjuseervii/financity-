from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# ✅ IMPORTANT: CORS block hatao taaki Express aur React dono clear access payein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models safely load ho rahe hain
model = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_category_predictor.pkl'))
feature_encoders = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_encoders.pkl'))
target_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'target_encoder.pkl'))

# ✅ FIXED ROUTE: Dono /predict aur /predict/ ko handle karne ke liye dict format
@app.post("/predict")
@app.post("/predict/")
def predict_financial_path(user_data: dict):
    try:
        # Debug karne ke liye console par print karega ki kya data aaya
        print("📥 Received Data from Express:", user_data)
        
        # Express se jo key names aa rahe hain unhe exact nikalna
        age = user_data.get("Age")
        gender = user_data.get("Gender")
        qualification = user_data.get("Qualification")
        goal = user_data.get("Financial_Goal")

        # Fallback check agar data key name uppercase/lowercase miss ho jaye
        if age is None:
            age = user_data.get("age")
            gender = user_data.get("gender")
            qualification = user_data.get("qualification")
            goal = user_data.get("financialGoal")

        input_dict = {
            'Age': [int(age)],
            'Gender': [str(gender)],
            'Qualification': [str(qualification)],
            'Financial_Goal': [str(goal)]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        for col in ['Gender', 'Qualification', 'Financial_Goal']:
            encoder = feature_encoders[col]
            input_df[col] = encoder.transform(input_df[col])
        
        predicted_code = model.predict(input_df)[0]
        final_recommended_path = target_encoder.inverse_transform([predicted_code])[0]
        
        return {
            "status": "success",
            "recommended_path": final_recommended_path,
            "demographics": {
                "age": age,
                "qualification": qualification,
                "goal": goal
            }
        }
    except Exception as e:
        print("❌ ML Engine Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))