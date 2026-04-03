from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle

app = FastAPI()

with open("kmeans.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    a: float
    b: float

@app.get("/")
def home():
    return {"message": "KMeans API running"}

@app.post("/predict")
def predict(data: InputData):
    point = np.array([[data.a, data.b]])
    cluster = model.predict(point)[0]
    center = model.cluster_centers_[cluster]
    return {
    "cluster": int(cluster),
    "center": center.tolist()
}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



