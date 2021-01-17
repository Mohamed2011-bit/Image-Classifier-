import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import json

def classifier_builder(size_input, size_hidden, size_output, dropout):
    sizes = [size_input] + size_hidden + [size_output]
    list_of_layer = []
    
    for i in range(1, len(sizes)):
        list_of_layer.append(('fc' + str(i), torch.nn.Linear(sizes[i - 1], sizes[i])))
        if i < len(sizes) - 1:
            list_of_layer.append(('relu' + str(i), torch.nn.ReLU()))
            list_of_layer.append(('drop' + str(i), torch.nn.Dropout(dropout)))
            
    list_of_layer.append(('output', torch.nn.LogSoftmax(dim=1)))
    classifier = torch.nn.Sequential(OrderedDict(list_of_layer))
    
    return classifier



def model_creation(arch):
    model = eval('models.' +  arch + '(pretrained=True)')

    for param in model.parameters():
        param.requires_grad = False
    
    return model


size_input = 25088 
size_hidden = [] 
size_output = 102
dropout = 0

arch = 'vgg13'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
model = model_creation(arch)
model.classifier = build_classifier(size_input, size_hidden, size_output, dropout)
model.to(device)


def validation(model, dataloader, criterion, device):
    model.to(device)

    accuracy = 0
    loss_validation = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images, labels = imagesinputs.to(device), labels.to(device)
        
        result = model.forward(images)
        loss_validation += criterion(result, labels).item()

        ps = torch.exp(result)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss_validation, accuracy


def train_classifier(epochs, print_every, model, train_data, valid_data, optimizer, criterion, device):
    model.to(device)
    moves = 0
    
    for e in range(epochs):
        losses = 0
        model.train()
        for i, (images, labels) in enumerate(train_data):
            moves += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if moves % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_data, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(loss_so_far/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))

                loss_so_far = 0

                model.train()
                

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.01)



def accuracy_check(model, dataloader):    
    accurate = 0
    total = 0
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            result = model(images)
            _, predicted = torch.max(result.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()

    print('Network accuracy on 10000 test images: %d %%' % (100 * accurate / total))
    


model.class_to_idx = image_datasets['train_data'].class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epoch': epochs,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer': optimizer.state_dict()
              }

torch.save(checkpoint, 'checkpoint.pth')


def checkpoint_loading(filename):
    
    check = torch.load(filename)
    model = models.vgg13(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = check['class_to_idx']
    classifier = nn.Sequential(OrderedDict([ ('fc1', nn.Linear(25088, 2960,  bias=True)),
                                                ('Relu1', nn.ReLU()),
                                                ('Dropout1', nn.Dropout(p = 0.5)),
                                                ('fc2', nn.Linear(2960, 102,  bias=True)),
                                                ('output', nn.LogSoftmax(dim=1))
                                                 ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_processing = Image.open(image)
   
    prepoceess_img = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pymodel_img = prepoceess_img(img_processing)
    return pymodel_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

imshow(process_image("flowers/test/1/image_06764.jpg"))



def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    image_torch = process_image(image_path)
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()
    
    with torch.no_grad():
        output = model.forward(image_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)


def check_sanity(path, model):
    pros, labels = predict(path, model)

    processed_image = process_image(path)

    label_map = {v: k for k, v in model.class_to_idx.items()}

    classes = [cat_to_name[label_map[l]] for l in labels]

    title = cat_to_name[image_path.split('/')[-2]]

    f, (ax1, ax2) = plt.subplots(2, 1, figsize = (6,6))
    plt.tight_layout()

    imshow(processed_image, ax=ax1, title=title)

    ax1.set_xticks([])
    ax1.set_yticks([])

    class_ticks = np.arange(len(classes))
    ax2.barh(class_ticks, probs)
    ax2.invert_yaxis()
    ax2.set_yticks(class_ticks)
    ax2.set_yticklabels(classes)


path = train_dir + '/31/image_06917.jpg'

check_sanity(path, model)