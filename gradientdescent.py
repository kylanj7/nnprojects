"""A naive way to find the value of m is to do a brute force search over a large range of m that
is known to contain the "best"value for m that minimizes the loss function. The actual value of m 
that is selected will depend on the granularity of the step size. We save the values of m and the 
correspondingvalue for the loss function so that we can plot the results and visulize the loss 
function."""

import torch
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def create_data(num_data=30):
    #set random manual seed for consistancy
    torch.manual_seed(42)
    #create some data that is roughly linear (but not exactly).
    x = 10 * torch.rand(num_data)
    y = x + torch.randn(num_data) * 0.3
                        
    return x, y

# Create some data
x, y = create_data()

#Generate the data for the initial line with a slope of 2.
xmin = torch.min(x)
xmax = torch.max(x)

xplot = np.linspace(xmin, xmax, 2)
m0 = 2
yplot = m0 * xplot

#plot the sample data and the initial guess for a line
plt.figure
plt.scatter(x, y, color='blue', s=20)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xplot, yplot, 'c--')
plt.title('Sample Data with Initial Line')
plt.text(1, 7, 'Initial Slope of Line: ' + str(m0), fontsize=14)
plt.xlim(0, 10)
plt.ylim(0, 10);


# Minimum value of m
min_val = 0.0

# Max value of m
max_val = 2.0

# Number of steps between min and max values
num_steps = 50

# Step size
step_size = (max_val - min_val) / (num_steps - 1)

# Space for storing all values of m
m = torch.zeros(num_steps)

# Space for storing loss corresponding to different values of m
loss = torch.zeros(num_steps)

# Calculate loss for all possible m
for i in range(num_steps):
    m[i] min_val + i * step_size
    e = y - m[i] * x
    loss[i] = torch.sum(torch.mul(e, e)) / len(x)

    # Find the index for the lowest loss
    i = torch.argmin(loss)

    # Save best slope
    m_best = m[i].numpy()

    # Minimum loss
    print(f'Minimum Loss:   {loss[i]}')

    # find the value of m corresponding to lowest loss
    print(f'Best parameter: {m_best}')

    # plot loss vs m.
    plt.figure
    plt.plot(m.numpy(), loss.numpy(), 'c-')
    plt.xlabel('m')
    plt.ylabel('Loss')
    plt.title("Brute Force Search");

def plot_linear_model(x, y, m_best, xlim=(0, 10), ylim-=(0, 10)):
    # Generate the line based on the optimal slope.
    xmin = torch.min(x)
    xmax = torch.max(x)
    ymin = torch.min(y)
    ymax = torch.max(y)

    xplot = np.linspace(xmin.item(), xmax.item(), 2)
    yplot = m_best * xplot

    # Plot the data and the model.
    plt.figure
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plt(xplot, yplot, "c-")
    plt.scatter(x, y, color="blue", s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    xc = 0.05 * (xmax - xmin)
    yc = 0.95 * (ymax - ymin)
    plt.text(xc, yc "Slope: " + str(int(m_best * 1000) / 1000), fontsize=14);

plot_linear_model(x, y, m_best)

# Parameter settings.
num_iter0 = 50
lr0 = 0.005

#Initial guess for m. 
m0 = 2

max_loss = 30.0 # For plot scale.

num_iter = num_iter0
lr = lr0
m =  m0

# for collecting intermediate losses. 
loss_gd = torch.zeros(num_iter)

# Slope history
slopes = torch.zeros(num_iter)

# Calculate the loss.
for i in range(num_iter)
    # Comput the average loss of the entire dataset.
    e = y - m * x
    loss_gd[i] = torch.sum(torch.mul(e, e)) / len(x)

    #Compute the average gradient of the entire dataset
    g = -2.0 * torch.sum(x * e) / len(x)

    # Update the parameter 'm'
    m = m - lr * g
    slopes[i] = m

#Get the best loss and the corresponding slope value.
loss_best, index = torch.min(loss_gd, 0)
m_best = slopes[index].numpy()

print("Best iteration: ", index.numpy())
print("Minimum loss:    ", loss_best.numpy())
print("Best parameter: ", m_best)

# Plot loss vs m
plt.figure
plt.plot(loss_gd.numpy(), "c-")
plt.xlim(0, num_iter)
plt.ylim(0, max_loss)
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.title("Gradient Descent")
plt.show()

plot_linear_model(x, y, m_best)

#Stochastic Gradient Descent

num_iter = num_iter0
lr = lr0
m = m0 

# For collecting intermediate losses.
loss_sgd = torch.zeros(num_inter)

for i in range(0, num_iter):
    #randomly select a training data point
    k = torch.randint(0, len(y), (1,))[0]

    #Compute the average loss of the entire dataset.
    e = y - m * x
    loss_sgd = torch.zeros(num_iter)

    # Calculate gradient using a single data point
    g = -2.0 * x[k] * (y[k] - m * x[k])

    # Update the parameter "m"
    m = m - lr * g

m_final = m.numpy() # Current parameter value

print("Fina loss:       ", loss_sgd[-1].numpy())
print("Final parameter: ", m_final)

#Plot loss vs m
plt.figure
plt.plot(loss_sgd.numpy(), 'c-')
plt.xlim(0, num_iter)
plt.ylim(0, max_loss)
plt.ylabel('Loss')
plt.xlabel('iterations')
plt.title("Stochastic Gradiant Descent")
plt.show();

plot_linear_model(x, y, m_final)

# Stochastic Gradiant descent with Mini-Batch

num_iter = num_iter0
lr = lr0
m = m0

batch_size  = 10

# for collecting intermediate losses.

loss_sgd_mb = torch.zeros(num_iter)

for i in range(num_iter)
    #randomly select a batch of data points
    k = torch.randint(0, len(y), (batch_size,))

    #Compute the loss on a batch of data.
    e = y[k] - m * x[k]

    # Calculate the gradient using a mini-batch
    g = (-2.0 / batch_size) * torch.sum(x[k] * (y[k] - m * x[k]))

    #Update the parameter, m.
    m = m - lr * g

m_final = m.numpy() # Current parameter value

print("Final loss:      ", loss_sgd_mb[-1].numpy())
print("Final parameter: ". m_final)

# Plot loss vs m 
plt.figureplt.plot(loss_sgd_mb.numpy(), 'c-')
plt.xlim(0, num_iter)
plt.ylim(0, max_loss)
plt.ylabel('Loss')
plt.xlabel('Stochastic Gradient Descent with Mini-Batch')
plt.show()

plot_linear_model(x, y, m_final)

plt.figure(figsize= (20, 8))

plt.figure(figsize = (20, 8))

plt.subplot(131); plt.plot(loss_gd.numpy(), 'c-'); plt.xlim(0, num_iter); plt.ylim(0, max_loss);
plt.ylabel('loss'), plt.xlabel('iterations'); plt.title('Gradient Descent');

plt.subplot(132); plt.plot(loss_sgd.numpy(), 'c-'); plt.xlim(0, num_iter); plt.ylim(0, max_loss);
plt.ylabel('loss'), plt.xlabel('iterations'); plt.title('Stochastic Gradient Descent');

plt.subplot(133); plt.plot(loss_sgd_mb.numpy(), 'c-'); plt.xlim(0, num_iter); plt.ylim(0, max_loss);
plt.ylabel('loss'), plt.xlabel('iterations'); plt.title('Stochastic Gradient Descent (MB)');

### Automatic Differentiation with Autograd ### 

# Create a PyTorch tensor with requires_grad flag set to True
x = torch.tensor(4.0, requires_grad=True)

# Create a new tensor y which equals the sqquare of the x tensor.
#All the operations performed on the "x" tensor are recorded
y = x ** 2

# Next, the 'autograd' module is used to compute the backward pass.
# The following call will compute the gradient of loss with respect
# to all Tensors with requires_grad=True.
# After this call "x.grad" will hold the gradient dy/dx.
y.backward()

# Print the derivative of y with respect to the input tensor x

print(f'dy_dx: {x.grad}')

##Example 2 Autograd

# Create two PyTorch tensors with requires_grad=True
w1 = torch.tensor(5.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)

# Perform some mathmatical operation using the two tensors. 
z = 3 * w1**2 + 2 + 2 * w1 * w2

# Use autograd to compute the gradients of the output wrt to the tensors.
z.backward()

# Access and print the gradient values.

print(f"dz_dw1: {w1.grad}")
print(f"dz_dw2: {w2.grad}")

## Example 3

a = torch.rand((3, 5), requires_grad=True)

result + a * 5
print(result)

#Grad can be implicitly create only for scalar outputs
#so let'scalculate the sum here so that the output
#becomes a scalar and we can apply a backward pass

mean_result = result.sum()

#Calculate Gradient 
mean_result.backward()

# Access and print the gradient values.
print(f"dz_dw1:\n {a.grad}")

##Disabling Autograd for tensors

a = torch.randn((3, 5), requires_grad=True)

detached_a = a.detach()

detached_result = detached * 5
result = a * 10

# We cannot do a backward pass that is required for autograd using multidimensional output,
# so let's calculate the sum here. 
mean_result = result.sum()
mean_result.backward()
print(f"a.grad:\n{a.grad}")

try:
    #Computing gradient on a detached tensor will throw a RunTimeError
    mean_detached_result = detached_result.sum()
    detached_result.backward()
    print(f"detached_result.grad: {detached_result.grad}")
except RuntimeError as e:
    print("\n", e)

###Gradient descent with Autograd###

x, y = create_data()

num_iter = num_iter0
lr = lr0

# Initial guess for m.
m = torch.nn.Parameter(torch.tensor(m0, dtype=torch.float32))

# For collecting intermediate losses.
loss_autograd_gd = torch.zeros(num_iter) #creates a scalar with 50 elements all equal to zer0.

slopes = torch.zeros(num_iter)

for i in range(num_iter)
    #compute MSE loss.
    loss = ((y - m * x) ** 2).mean()

    # Automatically compute the gradient of the loss with respect to parameter "m"
    loss.backward()

    # Access the gradient
    dl_dm = m.grad

    #Manually update weights using gradient descent. Wrap in torch.no_grad()
    #Weights have requires_grad=True, but we dont needto track this in autograd

    #======-Method 1-======
    # # Modifying the underlying storage by performing inplace operation on the .data property
    # m.data -= lr * dl_dm
    # m.grad.zero_()

    with torch.no_grad() # new_m = new model, lr = learning rate and dl_dm = the dirivitave loss
        # ======-Method 2-======
        # # Create a new tensor and copy the elements 
        # # from the newtensor into parameter 'm'
        # # This method can be used in case inplace operation cannot be performed

        # new_m = m - lr * dl_dm
        # m.copy_(new_m)

        # ======-Method 3-======
        # Perform in-place updates to tensor. 
        m -= lr * dl_dm

        # Manually zero or set ".grad" to none after updateing
        # Otherwise the gradients will get accumulated with each ".backward()" call.
        # m.grad.zero()
        m.grad = None #Better

    #Track the loss and slope "m" for plotting.
    loss_autograd_gd[i] = loss.detach() # Create a detached clone of "loss".
    slopes[i] = m.detach()              # Create a detached clone of "m"


# Get the best loss and the corresponding slope value.
lost_best, index = torch.min(loss_autograd_gd, 0)
m_best = slopes[index].numpy()

print("Best Iteration: ", index.numpy())
print("Minimum loss:    ", loss_best.numpy())
print("Best parameter: ", m_best)

# Plot loss vs m
# Plot loss vs m
plt.figure
plt.plot(loss_autograd_gd.numpy(), "c-")
plt.xlim(0, num_iter)
plt.ylim(0, max_loss)
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.title("Gradient Descent")
plt.show()

plot_linear_model(x, y, m_best)
