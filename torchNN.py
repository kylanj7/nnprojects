import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as fransforms
import matplotlib.pyplot as ply

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-Parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_sizes = 100
learning_rate = 0.001

MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor()
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,)
