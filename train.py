import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import iutils
import argparse
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

arg_parse = argparse.ArgumentParser(description='Train.py')

arg_parse.add_argument('data_dir', action="store", default="./flowers/")
arg_parse.add_argument('--gpu', dest="gpu", action="store", default="gpu")
arg_parse.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
arg_parse.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01)
arg_parse.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
arg_parse.add_argument('--epochs', dest="epochs", action="store", type=int, default=20)
arg_parse.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str)
arg_parse.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)

parse_args = ap.parse_args()
root = parse_args.data_dir
path = parse_args.save_dir
learning_rate = parse_args.learning_rate
structure = parse_args.arch
dropout = parse_args.dropout
hidden_layer1 = parse_args.hidden_units
device = parse_args.gpu
epochs = parse_args.epochs

def main():
    trainloader, v_loader, testloader = iutils.load_data(root)
    model, optimizer, criterion = iutils.network_construct(structure,dropout,hidden_layer1,learning_rate,device)
    iutils.do_deep_learning(model, optimizer, criterion, epochs, 40, trainloader, device)
    iutils.save_checkpoint(model,path,structure,hidden_layer1,dropout,learning_rate)
    print("Successfull Training!")

if __name__== "__main__":
    main()