import shutil
from typing import List
from fastapi import FastAPI,UploadFile, File
from features.main_measure import main
import os
app = FastAPI()



@app.post("/img")
async def root(files:List[UploadFile] = File(...)):
    for img in files:
        with open(f'../data/processed/{img.filename}','wb') as buffer:
            shutil.copyfileobj(img.file,buffer)
            res = main(img.filename)
    print(res)
    return {"res": res}
