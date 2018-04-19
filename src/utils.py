import pandas as pd
import numpy as np
import keras

def read_data(path):
    df = pd.read_csv(path, header=None)
    Xs = []
    Ys = []

    for id, group in df.groupby(0):
        X = group.values[None, :,2:-1]
        Y = group.values[0, -1]
        Xs.append(X)
        Ys.append(Y)

    Xs = np.concatenate(Xs)
    Ys = keras.utils.to_categorical(np.asarray(Ys))
    return (Xs, Ys)