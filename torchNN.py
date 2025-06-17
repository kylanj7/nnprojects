"""
    A typical PyTorch pipeline looks like this:
        1. Design model(input, output, forward pass with different layers)
        2. Construct loss and optimizer
        3. training loop
            - Forward = compute prediction loss
            - Backward = compute gradients
            - Update weights 
"""


# Utilize GPU, Datasets, DataLoader, Transforms, Neural Net, Training & Evaluation

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as fransforms
import matplotlib.pyplot as ply

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_sizes = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show

# Fully connected neural network with one hidden layer
class NeuralNet(nn.module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.11 = nn.Linear(input_size, hidden_size)
        self.ReLU()
        self.12 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.11(x)
        out = self.relu(out)
        out = self.12(out)
        # no activation and no softmax at the end
        return out
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Orgin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+l) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+l}/{n_total_steps}], Loss: {loss.item():.4f}')
