from keras import metrics
import keras.backend as K
from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.merge import *
from keras.layers.convolutional import *
from keras.layers.embeddings import *
from keras.layers.wrappers import *
from keras.layers.normalization import *
from keras.layers.advanced_activations import *
from keras.optimizers import *
from keras.models import *
from keras.callbacks import *
from keras.utils.vis_utils import *



def train(x_train, y_train, x_val, y_val, opt):
    n, length_bin, num_marks = x_train.shape
    _, num_label = y_train.shape
    
    run_name = opt.run_name
    
    input_shape = (length_bin, num_marks)
    input = Input(shape=input_shape)

    nn = Conv1D(50, 10, padding='same', activation="relu")(input)
    nn = MaxPooling1D(5, strides=5, padding='same')(nn)
    nn = Dropout(0.5)(nn)
    nn = Flatten()(nn)
    nn = Dense(625, activation="relu")(nn)
    nn = Dense(125, activation="relu")(nn)
    nn = Dense(num_label, activation="softmax")(nn)


    clf = Model(inputs=input, outputs=nn)
    clf.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["acc"])
    

    tb = TensorBoard(log_dir=f"../log/{run_name}", batch_size=32, write_graph=True)
    checkpointer = ModelCheckpoint(filepath=f"../model/{run_name}.h5", verbose=1, save_best_only=True, monitor="val_acc")

    clf.fit(
        x_train,
        y_train,
        batch_size=2000,
        callbacks = [tb, checkpointer],
        epochs=opt.epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
        verbose=0)
    

    return clf













    
    
    






    
    
    
