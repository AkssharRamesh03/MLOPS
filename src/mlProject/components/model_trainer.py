import pandas as pd
import os
from mlProject import logger
import joblib
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
import tensorflow
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        ann = Sequential()
        ann.add(Dense(64, activation='relu', input_shape=(train_x.shape[1],)))
        ann.add(Dense(32, activation='relu', input_shape=(train_x.shape[1] - 2,)))
        ann.add(Dense(10, activation='softmax'))
        loss = keras.losses.sparse_categorical_crossentropy
        optimizer = Adam(lr=0.01)
        ann.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        ann.fit(train_x, train_y, epochs=5)

        joblib.dump(ann, os.path.join(self.config.root_dir, self.config.model_name))

