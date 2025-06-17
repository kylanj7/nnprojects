import torch

#Tensors, everything in pytorch is based on tensors. 
#a tensor is a multi-dimensional matrix containing elements of a single data type

#.empty() initializes the tensor without setting the values. 
#code below describes how to initialize different types of Tensors. 

x = torch.empty(1) #scalar (single element) ( think dot on a piece of paper)
print(x)

x = torch.empty(3) #3 point vetor (think single line on paper)
print(x)

x = torch.empty(2,3) #Matrix (two perpendicular , think drawing a square on a page)
print(x)

x = torch.empty(2,2,3) #3rd Dimensional Tensor (cube)
print(x)

x = torch.empty(2,2,2,3) #4th Dimensional Tensor (tesseract)
print(x)

#creates a tensor of specified size with random numbers [0-0.99]
x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3) #fills with Zero's
print(x)

x = torch.ones(5,3)#fills with 1's
print(X)

print(x.size()) #prints the size of the tensor
print(x.shape) #prints the shape of the tensor

#check the data type
print(x.dtype)

#specify types, float32 default
x = torch.zeros(5,3, dtype=torch.float16)
print(x)


x = torch.tensor([5.5,3]) 
#contruct from different data, insert desired data 
#to form tensor to the users specifications
print(x)

#Requires_grad argument
#This will tell pytorch that it will need to calculate the gradients for this 
# tensor later in your optimizations steps. i.e this is a variable 
# in your model that you want to optimize
x = torch.tensors([5.5,3], requires_grad=True)
print(x)

###OPERATIONS with TENSORS###

x = torch.ones(2,2)
y = torch.ones(2,2)

# elementwise additions z = x + y

torch.add(x,y) 

# in place addition, everything with trailing underscore is an inplace operation
#i.i it will modify the variable
#y.add_(x)

print(x)
print(y)
print(z)

#subtraction z = x - y

z = torch.sub(x,y)

# multiplication z = x * y

z = torch.mul(x,y)

# Division z = x / y

z = torch.div(x,y)

# Slicing [] index 

x = torch.rand(5,3)

print(x)
print(x[:,0]) # Slices all rows, starts at column index 0
print(x[1,:]) # Starts at row 1, slices all columns after. 
print(x[1,1]) # elemet at column 1, row 1

# Get the actual value of only 1 element that is in your tensor
print(x[1,1].item())

#Reshape the tensor with torch.view()
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) # the size -1 is inferred from other dimensions
# if -1 PyTorch will automatically determine the necessary size.
print(x.size(), y.size(), z.size())

## Convert Tensor to NumPy Array##

a = torch.ones(5) # creates a tensor vector length of 5 filled 
#with ones
print(a)

#torch to numpy with numpy
b = a.numpy()
print(b)
print(type(b))

#If tensor in on the CPU, both objects will share the same memory location
#changing one will also change the other

a.add_(1)
print(a)
print(b)

#NumPy to PyTorch w/ .from_numpy(x), or torch.tensor() to copy it.
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
c = torch.tensor(a)
print(a)
print(b)
print(c)

# Again be careful when modifying

a += 1
print(a)
print(b)
print(c)

##GPU Support.##
##All tensors are created on the CPU. We can also moce them to the 
# GPU. 
##Alternativly, we can create them directly from the GPU.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# GPUs can be selected by indexing them with :0 for the first and :1 
# for the 2nd & so on

x = torch.rand(2,2).to(device) # Move tensors to GPU device
#x = x.to("cpu")
#x = x.to("cuda")

x = torch.rand(2,2, device=device) # Create tensor directly create 
#them on GPU, more efficient

##AUTOGRAD##

#Autograd is a Python Library that automatically differentiates native
# Python and NumPy Code. It is designed to handle a large subset of 
# Python's features, including loops, if statements, recursion and 
# closures. Autograd supports both reverse-mode differentiation 
# (back-prop)and forward-mode differentiation,
# making it highly versitile for gradient-based optimization tasks. 

# Provides automatic differentiation for all operations on tensors. 
# Generally speaking, torch.autograd is an engine for computing the 
# Vector Jacobian product. It computes partial dirivates while 
# applying the chain rule. 

#requires_grad = True -> tracks all operations on the tensor with a computational graph
#We create this when we want to know the gradients with respect to x
x = torch.randn(3, requires_grad=True) 
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute.
# grad_fn: references a Function that has created the Tensor
print(x) # created by the user -> grad_fn is None
print(y)
print(y.grad_fn)


# requires_grad stays True for all operations 
# involving Tensor 'x'

z = y * y * 3
print(z)

z = z.mean()
print(z)

z.backward() # perform back propogation
print(x.grad) # returns dz/dx. must perform z.backwards first. 

# dz/dx represents the rate of change of the variable z 
# (often a loss or output)in relation to x (often the input). 
# In the context of a neural net, z could represent the output 
# of a layer, while x could be an input feature or weight where 
# 'd' represents the derivitave of each respective value after
# the mathmatical operations have been made.

#!!! backward() accumulates the gradient for the tensor in .grad attribute.
#!!! We need to be careful during optimization !!! optimizer.zero_grad

# Sometimes you do not want to track the gradient,
# stop a tensor from being tracked: During the training loop 
# when we want to evaluate our weights, or after training during 
# evaluation. These operations should not be 
# part of the gradient computation


# .requires_grad_(...) changes an existing flag in-place
a = torch.randn(2,2)
b = (a * a).sum()
print(a.requires_grad)
print(b.grad_fn)

a.requires_grad_(True)
b = (a * a).sum()
print(a.requires_grad)
print(b.grad_fn)

# .detach(): get a new Tensor with the same content but no gradient computation:
a = torch.randn(2,2,requires_grad=True) # adds the gradient tracking flag
b = a.detach() # removes the gradient tracking flag
print(a.requires_grad) # returns True
print(b.requires_grad) # return False

# wrap in 'with torch.no_grad()
a = torch.randn(2,2, requires_grad=True)
print(a.requires_grad)
with tirch.no_grad():
    b = a ** 2
    print(b.requires_grad)

##GRADIENT DESCENT## LINEAR REGRESSION###

# f(x) = weights * inputs + bias
# f(x) = w * x + b

#approximate a simple function: f(x) = 2 * x (ignore bias for this example)

X = torch.tensor([1,2,3,4,5,6,7,8], dtype=torch.float32)  
Y = torch.tensor([2,4,6,8,10,12,14,16], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) #(Initialize weight value, datatype, gradient tracking)

#model output
def forward(x): 
    return w * X

# loss = MSE (Mean Squared Error)
def loss(y, y_pred): #y_pred becomes the forward pass w * x
    return ((y_pred - y)**2).mean() 
# take the average after subtracting the predictions from the 
# actual value, then rasing to the power of 2

X_test = 5.0 #testing the forward pass with 5.0 as the input

print(f'Prediction before training: f({X_test}) - {forward(X_test).item():.3f}')

#currently the weights are set to 0.0 because we have yet to find meaningful 
#training data. In this section, we begin to understand the training procedure

# Training
learing_rate = 0.01
n_epochs = 100

for epoch in range(n_epochs)
    #predict = forward pass
    y_pred = forward(X)

    #loss calculation
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    #update weights stop tracking gradiants 
    #with this calculation
    #w.data = w.data - learning rate * w.grad 
    with torch.no_grad():
        w -= learning_rate * w.grad

    #zero the gradients after updating
    w.grad.zero_()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.3f}')

print(f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')

### Model, Loss and Optimizer ###

import torch.nn as nn

#Linear regression
#f = w * x
# For this example: f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16]], dtype=torch.float32)

x = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
        return w * x

# loss = MSE
def loss(y, y_pred):
        return ((y_pred - y)**2).mean()

X_test = 5.0

print(f'Prediction before training: f({X_test}) = {forward(X_test).itme():.f}')

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# Design the model, the model has to implement the forward 
# pass!
# Here we could simply usea built-in model from PyTorch
# Model = nn.linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define different layers
        self.lin = nn.linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
input_size, output_size = n_features, n_features


print(f'Prediction before training: ', ({X_test.item()} = {model(X_test).item():.3f}))

##MODEL LOSS and OPTIMIZER###

# 2) Define loss and optimizer
learning_rate = 0.1 #this is a hyper parameter
n_epochs = 100 #epoch represents an etire pass in the machine learning dataset.

loss = nn.MSELoss()4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#This optimizer uses Stochastic Gradient descent. It's a more efficient optimization algorithm 
#than standard gradient descent. 

# 3) Training loop
for epoch in range(n_epochs):
    # predict = forward pass with our model
    y_predicted = model(X)

    # Loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # Update the weights
    optimizer.zero_grad()

    # zero the gradients after updating
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
         w, b = model.parameters() # unpack parameters
         print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')

