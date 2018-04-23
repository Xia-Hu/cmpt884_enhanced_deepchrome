from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers


def train(x_train, y_train, opt):
    ### parameter setup
    size_batch = 1
    num_filters = 50
    length_filter = 10
    length_pool = 5
    length_full_1 = 625
    length_full_2 = 125
    lrate=0.01

    ### construct model
    model = Sequential()
    # conv + relu + pooling
    model.add(Convolution2D(
        batch_input_shape=(size_batch, 1, opt.num_marks, opt.length_bin),
        filters=num_filters,
        kernel_size = (opt.num_marks, length_filter),
        strides=1,
        padding='same',
        data_format='channels_first'
        ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(1, length_pool),
        strides=length_pool,
        padding='same',
        data_format='channels_first'
        ))
    # dropout
    model.add(Dropout(0.5))
    model.add(Flatten())
    # two hidden layer
    model.add(Dense(length_full_1))
    model.add(Activation('relu'))
    model.add(Dense(length_full_2))
    model.add(Activation('relu'))
    # output layer
    model.add(Dense(opt.num_label))
    model.add(Activation('softmax'))
    
    # loss and gd
    #sgd = optimizers.SGD(lr=lrate)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    ### begin training
    model.fit(x_train, y_train, epochs = opt.enhancedepoch, batch_size = 100)

    return model













    
    
    


