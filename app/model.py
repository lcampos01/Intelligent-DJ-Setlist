import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from config import FEATURES_CSV, MODEL_PATH
from utils import to_camelot, camelot_distance, bpm_penalty, compute_blending_score

def train_model():
    df = pd.read_csv(FEATURES_CSV)

    # Añadir columna Camelot
    df["camelot"] = df["key"].apply(to_camelot)

    features = [c for c in df.columns if c not in ["filename", "key", "camelot", 'bpm']]
    X = df[features].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = min(10, max(1, len(df) - 1))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(Xs)

    joblib.dump((df, scaler, knn), MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return knn

def recommend(filename, top_n=5):
    df, scaler, knn = joblib.load(MODEL_PATH)

    if filename not in df["filename"].values:
        raise ValueError(f"La canción '{filename}' no existe en el dataset.")

    features = [c for c in df.columns if c not in ["filename", "key", "camelot", 'bpm']]
    row = df[df["filename"] == filename]
    x = scaler.transform(row[features].values)

    n_neighbors = min(top_n + 1, len(df))
    distances, indices = knn.kneighbors(x, n_neighbors=n_neighbors)
    idx = indices[0][1:]

    query_bpm = float(row["bpm"].values[0])
    query_camelot = row["camelot"].values[0]

    recs = []
    for i, d in zip(idx, distances[0][1:]):
        target = df.iloc[i]
        bpm_p = bpm_penalty(query_bpm, target["bpm"])
        camelot_p = camelot_distance(query_camelot, target["camelot"])
        score = compute_blending_score(d, camelot_p, bpm_p)
        recs.append({
            "filename": target["filename"],
            "distance": float(d),
            "bpm": float(target["bpm"]),
            "key": target["key"],
            "camelot": target["camelot"],
            "score": round(score, 4)
        })

    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs
