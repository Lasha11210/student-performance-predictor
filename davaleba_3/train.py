import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples = 1000
study_hours = np.random.uniform(5, 30, n_samples)
attendance = np.random.uniform(60, 100, n_samples)
assignments = np.random.randint(5, 11, n_samples)
midterm_score = np.random.uniform(40, 100, n_samples)

final_score = (
    0.5 * study_hours
    + 0.3 * attendance
    + 2.0 * assignments
    + 0.4 * midterm_score
    + np.random.normal(0, 3, n_samples)
)
final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame(
    {
        "study_hours": study_hours,
        "attendance": attendance,
        "assignments": assignments,
        "midterm_score": midterm_score,
        "final_score": final_score,
    }
)
df.to_csv("student_data.csv", index=False)

X = df[["study_hours", "attendance", "assignments", "midterm_score"]]
y = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Student_Performance_Analysis")

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("train_size", 0.8)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2_Score", r2)

    mlflow.sklearn.log_model(model, "student_score_model")

    active_run_id = run.info.run_id

print("-" * 50)
print(f"მოდელი წარმატებით გაიწვრთნა და დალოგდა MLflow-ში!")
print(f"Run ID: {active_run_id}")
print(f"MAE (საშუალო აბსოლუტური ცდომილება): {mae:.2f}")
print(f"R2 Score (სიზუსტე): {r2:.2f}")
print("-" * 50)