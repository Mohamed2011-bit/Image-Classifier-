import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch
import json
import PIL
import argparse
import iutils
from PIL import Image
from torch import nn
from torch import tensor
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from collections import OrderedDict

arg_parse = argparse.ArgumentParser(description='Predict.py')

arg_parse.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
arg_parse.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
arg_parse.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
arg_parse.add_argument('--top_k', default=3, dest="top_k", action="store", type=int)
arg_parse.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
arg_parse.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parse_args = ap.parse_args()
path_image = parse_args.input
number_of_outputs = parse_args.top_k
device = parse_args.gpu
path = parse_argsa.checkpoint

def main():
    model=iutils.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probabilities = iutils.predict(path_image, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Predicted Successfully!")

    
if __name__== "__main__":
    main()