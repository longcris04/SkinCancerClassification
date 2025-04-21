from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from utils import m_infer, infer_1, infer_2
from settings import SERVER_IP, SERVER_PORT

app = FastAPI()

@app.post("/predict")
def predict(input: dict):
    predict = m_infer(input['image'], input['model_name'])
    return predict

@app.post("/predict1")
def predict(input: dict):
    predict = infer_1(input['image'], input['model_name'])
    print(predict)
    return JSONResponse(predict)

@app.post("/predict2")
def predict(input: dict):
    predict = infer_2(input['image'], input['model_name'])
    print(predict)
    return JSONResponse(predict)

if __name__ == "__main__":
    uvicorn.run(app, host= f'{SERVER_IP}', port=SERVER_PORT)