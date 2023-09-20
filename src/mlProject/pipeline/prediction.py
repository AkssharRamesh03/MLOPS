import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow
from tensorflow.keras.models import load_model



class PredictionPipeline:
    def __init__(self):
        self.model = load_model(Path('artifacts/model_trainer/model.h5'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return np.argmax(prediction, axis=1)