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
from keras.layers.recurrent import LSTM
from keras import optimizers


def LSTM_bin(inputs, bname):
    time_steps = 1
    lstm_units = 100
    
    # LSTM
    out_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True), name=bname)(inputs)
    # ATTENTION
    out_att = Permute((2, 1))(out_lstm)
    out_att = Dense(time_steps, activation='softmax')(out_att)
    out_att = Permute((2, 1), name='attention_vec'+bname)(out_att)
    out_att = multiply([out_lstm, out_att], name='attention_mul'+bname)
    
    return out_att

def LSTM_hm(inputs):
    time_steps = 5
    lstm_units = 200
    
    # LSTM
    out_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True), name='hmlstm')(inputs)
    # ATTENTION
    out_att = Permute((2, 1))(out_lstm)
    out_att = Dense(time_steps, activation='softmax')(out_att)
    out_att = Permute((2, 1), name='attention_vec')(out_att)
    out_att = multiply([out_lstm, out_att], name='attention_mul')
    
    return out_att


def train(x_train, y_train, opt):
    ### parameter setup
    lstm_units = opt.length_bin
    input_dim = opt.length_bin
    lrate=0.01
    time_steps=1

    ### construct model

    # bin-level LSTM + Attention
    inputs1 = Input(shape=(time_steps, input_dim))
    out_att1 = LSTM_bin(inputs1, 'binlstm1')
    inputs2 = Input(shape=(time_steps, input_dim))
    out_att2 = LSTM_bin(inputs2, 'binlstm2')
    inputs3 = Input(shape=(time_steps, input_dim))
    out_att3 = LSTM_bin(inputs3, 'binlstm3')
    inputs4 = Input(shape=(time_steps, input_dim))
    out_att4 = LSTM_bin(inputs4, 'binlstm4')
    inputs5 = Input(shape=(time_steps, input_dim))
    out_att5 = LSTM_bin(inputs5, 'binlstm5')
    
    # HM-level merge
    mmerge = concatenate([out_att1, out_att2, out_att3, out_att4, out_att5], axis=-1)
    
    # HM-level LSTM
    out_hmatt = LSTM_hm(mmerge)
    
    # output layer
    flat = Flatten()(out_hmatt)
    output = Dense(opt.num_label, activation='softmax')(flat)
    
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5], outputs=output )
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    from keras.utils import plot_model
    #plot_model(model, to_file='model.png')
    
    ### begin training
    input1 = x_train[:,0,:]
    input1 = input1[:,np.newaxis,:]
    input2 = x_train[:,1,:]
    input2 = input2[:,np.newaxis, :]
    input3 = x_train[:,2, :]
    input3 = input3[:,np.newaxis, :]
    input4 = x_train[:,3,:]
    input4 = input4[:, np.newaxis, :]
    input5 = x_train[:, 4, :]
    input5 = input5[:, np.newaxis, :]
        
    model.fit([input1, input2, input3, input4, input5], y_train, epochs = opt.attentiveepoch, batch_size = 100)
    
    return model


