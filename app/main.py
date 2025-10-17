from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from extractor import build_features_csv
from model import train_model, recommend
from config import FEATURES_CSV, MODEL_PATH

app = FastAPI(title="Track Blending API", version="2.0")

class TrainRequest(BaseModel):
    rebuild_features: bool = True

@app.post('/build-features')
async def api_build_features():
    df = build_features_csv()
    return {'status': 'features_built', 'count': len(df), 'path': FEATURES_CSV}

@app.post('/train')
async def api_train(req: TrainRequest):
    if req.rebuild_features:
        build_features_csv()
    train_model()
    return {'status': 'trained', 'model_path': MODEL_PATH}

@app.get('/recommend/{filename}')
async def api_recommend(filename: str, top_n: int = 5):
    try:
        recs = recommend(filename, top_n=top_n)
        
        # SOLUCIÓN: Acceder a los diccionarios por clave (nombre)
        return {'query': filename,
                'recommendations': [
                    {
                        'filename': r['filename'], 
                        'distance': r['distance'], 
                        'score': r['score'],
                        'bpm': r['bpm'],       # Se recomienda incluir también BPM y clave
                        'camelot': r['camelot'] # para que el resultado sea útil
                    } 
                    for r in recs
                ]}
    except Exception as e:
        # Esto capturaría cualquier ValueError de 'model.py' o el KeyError si no lo corrigieras.
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/')
async def root():
    return {'service': 'track-blending', 'status': 'ok'}
