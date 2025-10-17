import os
import numpy as np
import pandas as pd
import essentia.standard as es
from config import MUSIC_DIR, FEATURES_CSV

def extract_features(audio_path):
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()

    # BPM
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, _, _, _, _ = rhythm_extractor(audio)

    # Energy
    energy = float(np.sqrt(np.mean(audio ** 2)))

    # Key
    key_extractor = es.KeyExtractor()
    key, scale, strength = key_extractor(audio)
    key_str = f"{key}_{scale}"

    # MFCCs
    w = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc_extractor = es.MFCC()
    mfccs = []
    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc_extractor(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
    mfcc_mean = np.mean(mfccs, axis=0)

    return {
        "filename": os.path.basename(audio_path),
        "bpm": float(bpm),
        "energy": float(energy),
        "key": key_str,
        **{f"mfcc_{i}": float(mfcc_mean[i]) for i in range(len(mfcc_mean))}
    }

def build_features_csv():
    files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR)
             if f.lower().endswith((".mp3", ".wav", ".flac"))]
    if not files:
        raise RuntimeError("No hay archivos de audio en ./data/music")

    features = []
    for path in files:
        try:
            feat = extract_features(path)
            features.append(feat)
            print(f"✅ Procesado: {os.path.basename(path)}")
        except Exception as e:
            print(f"⚠️ Error en {path}: {e}")

    df = pd.DataFrame(features)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nGuardado dataset en {FEATURES_CSV}")
    return df
