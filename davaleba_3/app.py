import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Performance Predictor API",
    description="API სტუდენტის საბოლოო ქულის პროგნოზირებისთვის",
    version="1.0"
)

try:
    experiment = mlflow.get_experiment_by_name("Student_Performance_Analysis")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    last_run_id = runs.iloc[0].run_id
    model_uri = f"runs://{last_run_id}/student_score_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"წარმატებით ჩაიტვირთა მოდელი Run ID-დან: {last_run_id}")
except Exception as e:
    print("მოდელის ჩატვირთვისას დაფიქსირდა შეცდომა:", e)

class StudentData(BaseModel):
    study_hours: float = Field(..., description="კვირაში სწავლის საათები", ge=0, le=168)
    attendance: float = Field(..., description="დასწრების პროცენტი", ge=0, le=100)
    assignments: int = Field(..., description="შესრულებული დავალებები", ge=0)
    midterm_score: float = Field(..., description="შუალედური ქულა", ge=0, le=100)

@app.post("/predict", summary="საბოლოო ქულის პროგნოზირება")
def predict_score(data: StudentData):
    input_data = pd.DataFrame([{
        'study_hours': data.study_hours,
        'attendance': data.attendance,
        'assignments': data.assignments,
        'midterm_score': data.midterm_score
    }])
    
    prediction = model.predict(input_data)[0]
    final_pred = float(np.clip(prediction, 0, 100))
    
    return {
        "status": "success",
        "predicted_final_score": round(final_pred, 2),
        "input_features": data.dict()
    }

@app.get("/")
def root():
    return {"message": "Welcome to Student API. Append /docs to URL for Swagger UI."}