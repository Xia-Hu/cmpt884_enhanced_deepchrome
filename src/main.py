import numpy as np
from optparse import OptionParser
import train_deepchrome
from utils import read_data

def training(opt):
    # generate input data with size
    Xtrain, Ytrain = read_data("../data/toy/train.csv")
    Xvalid, Yvalid = read_data("../data/toy/valid.csv")
    
    # train deep chrome
    model = train_deepchrome.train(Xtrain, Ytrain, Xvalid, Yvalid, opt)

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--epochs", dest="epochs", default=10)
    parser.add_option("--run_name", dest="run_name")
    (options, args) = parser.parse_args()
    
    training(options)
training(options)
