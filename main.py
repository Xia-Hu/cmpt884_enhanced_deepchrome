import numpy as np
from optparse import OptionParser
import train_deepchrome
import train_attentivechrome
import keras

def training(opt):
    # generate input data with size
    x_train = np.random.random((opt.num_train, opt.num_marks, opt.length_bin))
    #x_train = np.random.random((opt.num_train, opt.num_marks, opt.length_bin))
    y_train = keras.utils.to_categorical(np.random.randint(
        opt.num_label, size=(opt.num_train, 1)), num_classes=opt.num_label)
    # train deep chrome
    #model_dchrome = train_deepchrome.train(x_train, y_train, opt)
    
    # train attentive chrome
    model_achrome = train_attentivechrome.train(x_train, y_train, opt)
    
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--numtrain", dest="num_train", default=2000)
    parser.add_option("--nummark", dest="num_marks", default=5)
    parser.add_option("--lenbin", dest="length_bin", default=100)
    parser.add_option("--nlabel", dest="num_label", default=2)
    (options, args) = parser.parse_args()
    
    training(options)
