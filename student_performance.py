import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
from fastapi import FastAPI
from pydantic import BaseModel



data = {
    'Exam1': np.random.randint(50, 100, 100),
    'Exam2': np.random.randint(50, 100, 100),
    'Exam3': np.random.randint(50, 100, 100),
    'Exam4': np.random.randint(50, 100, 100),
    'Exam5': np.random.randint(50, 100, 100),
    'Attendance': np.random.randint(60, 100, 100),
    'Participation': np.random.randint(0, 5, 100),
}
df = pd.DataFrame(data)
df['Improvement_Rate'] = (df['Exam5'] - df['Exam1']) / df['Exam1']


df['Exam6'] = df['Exam5'] + np.random.randint(-5, 10, 100)


df.fillna(df.mean(), inplace=True)



scaler = StandardScaler()
features = ['Exam1', 'Exam2', 'Exam3', 'Exam4', 'Exam5', 'Attendance', 'Participation', 'Improvement_Rate']
X = scaler.fit_transform(df[features])
y = df['Exam6']

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Training Complete - MSE: {mse:.2f}, R2 Score: {r2:.2f}")


app = FastAPI()


class StudentData(BaseModel):
    Exam1: float
    Exam2: float
    Exam3: float
    Exam4: float
    Exam5: float
    Attendance: float
    Participation: int


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.post("/predict")
def predict(data: StudentData):
    print("Received Data:", data.dict())  # Print the received data
    try:
        
        input_data = pd.DataFrame([data.dict()])
        input_data['Improvement_Rate'] = (input_data['Exam5'] - input_data['Exam1']) / input_data['Exam1']
        features = ['Exam1', 'Exam2', 'Exam3', 'Exam4', 'Exam5', 'Attendance', 'Participation', 'Improvement_Rate']
        normalized_data = scaler.transform(input_data[features])
        prediction = model.predict(normalized_data)
        return {"Predicted_Exam6_Score": prediction[0]}
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}



# Run using: `uvicorn filename:app --reload`
