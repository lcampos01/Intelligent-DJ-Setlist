import os

DATA_DIR = os.environ.get('DATA_DIR', '/data')
MUSIC_DIR = os.environ.get('MUSIC_DIR', '/music')
FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'model.joblib')


# parámetros
KNN_K = 20
TOP_N = 5
SAMPLE_RATE = 22050