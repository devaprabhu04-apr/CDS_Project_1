from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel, Field

# 1. Initialize FastAPI app
app = FastAPI(title="SVC Classification API", version="1.0")

# 2. Load the saved model and scaler
with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 3. Define the input data structure based on your features
# Use Field with validation alias to handle spaces in JSON keys
class ModelInput(BaseModel):
     Age: float
     Gender: float
     Total_Bilirubin: float = Field(validation_alias="Total Bilirubin")
     Direct_Bilirubin: float = Field(validation_alias="Direct Bilirubin")
     Alkaline_Phosphotase: float = Field(validation_alias="Alkaline Phosphotase")
     Alamine_Aminotransferase: float = Field(validation_alias="Alamine Aminotransferase")
     Aspartate_Aminotransferase: float = Field(validation_alias="Aspartate Aminotransferase")
     Total_Protiens: float = Field(validation_alias="Total Protiens")
     Albumin: float
     Albumin_and_Globulin_Ratio: float = Field(validation_alias="Albumin and Globulin Ratio")

@app.get("/")
def home():
    return {"message": "Welcome to the SVC Model Deployment API!"}

@app.post("/predict")
def predict_endpoint(input_data: ModelInput):
    # Convert incoming JSON data to a 2D numpy array
    input_array = np.array([[
        input_data.Age,
        input_data.Gender,
        input_data.Total_Bilirubin,
        input_data.Direct_Bilirubin,
        input_data.Alkaline_Phosphotase,
        input_data.Alamine_Aminotransferase,
        input_data.Aspartate_Aminotransferase,
        input_data.Total_Protiens,
        input_data.Albumin,
        input_data.Albumin_and_Globulin_Ratio,
    ]])

    # 4. Scale the input using the saved MinMaxScaler (scaler variable)
    scaled_input = scaler.transform(input_array)

    # 5. Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input) if hasattr(model, "predict_proba") else None

    # 6. Return the result
    return {
        "prediction": int(prediction[0]),
        "probability": float(np.max(prediction_proba)) if prediction_proba is not None else "Not available"
    }
# ... (your FastAPI app, routes, and model loading code stay right here) ...
import os
import uvicorn

if __name__ == "__main__":
    # Render assigns a dynamic port via environment variables; defaults to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
    
