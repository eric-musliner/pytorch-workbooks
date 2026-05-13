import torch
import numpy as np
from torch import nn  # building blocks for neural networks
import matplotlib.pyplot as plt
from pathlib import Path


def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions
    :return:
    :rtype:
    """
    plt.figure(figsize=[10, 7])

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="blue", s=4, label="training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="green", s=4, label="testing data")

    # Are there predictions
    if predictions is not None:
        # Plot predictions
        plt.scatter(test_data, predictions, c="red", s=4, label="predictions")

    # Show legend
    plt.legend(prop={"size": 14})
    plt.show()


def main():
    what_were_covering = {
        1: "data (prepare and load)",
        2: "build model",
        3: "fitting the model to data (training)",
        4: "making predictions and evaluating a model",
        5: "saving and reloading a model",
        6: "putting it all together",
    }

    # Create some known data using linear regression formula
    # use linear regression formula to make a straight line with known parameters

    # Create known parameters
    weight = 0.7  # (b)
    bias = 0.3  # (a)

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    print(X[:10])
    print(y[:10])
    print(len(X))
    print(len(y))

    # Splitting data in to training and test sets

    # Create a train/test split
    train_split = int(0.8 * len(X))
    print(train_split)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))

    plot_predictions(
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        predictions=None,
    )


# Build model
# Create linear regression model class
#
# What our model does;
# Start with random values (weight & bias)
# Look at training data and adjust the random values to better represent (or get closer to) the ideal values
# ( the weight and bias values we used to create the data)
# How does it do so?
# Through two main algorithms
# 1. Gradient descent
# 2. Backpropogation
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # Forward method to define computation in model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input (training) data
        :return:
        :rtype:
        """
        return self.weights * x + self.bias  # this is the linear regression formula


def test_model():
    # Checking the contents of our pytorch model
    # Create random seed
    torch.manual_seed(42)

    # Create instance of model
    model_0 = LinearRegressionModel()
    print(model_0)
    print(list(model_0.parameters()))
    print(model_0.state_dict())

    # Create a train/test split
    # Create known parameters
    weight = 0.7  # (b)
    bias = 0.3  # (a)

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))
    print(train_split)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # print(X_test)
    # Making predictions using torch.inferene_mode()
    # Check models predictive power
    # Let's see how well it predicts y_test based on X_test
    # When we pass data through our model, it's going to run through the forward method
    #
    with torch.inference_mode():
        y_preds = model_0(X_test)
    print(y_preds)

    plot_predictions(
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        predictions=y_preds,
    )

    # The whole idea of training is for a model to move from some unknown paramters to known parameters
    # move from a poor representation to a better representation
    # One way to measure how poor or how wrong your models predictions are is to use a loss function (cost function)
    #
    # Loss fucntion -a  function to measure how wrong your models predictions are to the ideal outputs. lower is better
    # Optimizer - takes into account the loss of a model and adjusts the model's parameters (eg. weight and bias)
    #
    # FOr pytorch, we need
    # * training loop
    # * testing loop

    # Setup a loss function
    loss_fn = nn.L1Loss()

    # Setup an optimizer (stochastic gradient descent) lr = most important hyperparameter you can set
    optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

    # Building training loop (and a testing loop)
    # A couple of things we need
    # 0. Loop through the data
    # 1. Forward pass ( this involves data moving through our models 'forward' function(s)) - forward propagation
    # 2. Calculate the loss (compare forward pass predictions to ground truth)
    # 3. Optimize zero grad
    # 4. Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model
    #                    with respect to the loss (back propagation)
    # 5. Optimizer step - use the optimizer to adjust our models parameters to try and improve the loss (gradient descent)

    #  An epoch is one loop through the data. This is a hyperparameter because we've set it ourselves
    epochs = 200

    # Tracking different values (experiments)
    epoch_count = []
    loss_values = []
    test_loss_values = []

    # 0. loop through data
    for epoch in range(epochs):
        # Set the model to training mode
        model_0.train()  # sets all parameters for require_gradients to true

        # 1. Forward pass
        y_pred = model_0(X_train)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation on the loss with respect to the parameters of the model
        loss.backward()

        # 5. Step the optimizer (perform gradient descent)
        optimizer.step()  # by default how the optimizer changes will accumulate through the loop...we have to zero them above in step 3 for the next iteration of the loop

        # Testing
        model_0.eval()  # turns off gradient tracking and other settings in the model not needed for evaluation/testing
        with torch.inference_mode():  # turns off gradient tracking
            # 1. Do forward pass
            test_pred = model_0(X_test)

            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch}, Loss: {loss}, Test Loss: {test_loss}")

    print(model_0.state_dict())
    with torch.inference_mode():
        y_preds = model_0(X_test)
    print(y_preds)

    plot_predictions(
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        predictions=y_preds,
    )

    # Plot loss curves

    plt.plot(
        epoch_count,
        np.array([t.detach().item() for t in loss_values]),
        label="Training Loss",
    )
    plt.plot(epoch_count, test_loss_values, label="Test Loss")
    plt.title("Training and Loss Curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Saving a model in Pytorch
    #
    # Three main methods
    # 1. torch.save() - save a pytorch object in pythons pickle format
    # 2. torch.load() - allows you to load a pytorch object
    # 3. torch.nn.Module.load_state_dict() - allows to load a model's saved state dict

    # Create directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Create model save path
    model_name = "01_pytorch_workflow_model_0.pth"
    model_save_path = MODEL_PATH / model_name

    # Save the model state dict
    torch.save(model_0.state_dict(), model_save_path)


def load_model():
    # Loading pytorch model
    # Saved model's state dict instead of full model
    # Create new instance of model class
    loaded_model_0 = LinearRegressionModel()
    loaded_model_0.load_state_dict(torch.load("models/01_pytorch_workflow_model_0.pth"))
    print(loaded_model_0.state_dict())

    # Make some predictions with loaded model

    # Create a train/test split
    # Create known parameters
    weight = 0.7  # (b)
    bias = 0.3  # (a)

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))
    print(train_split)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    loaded_model_0.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model_0(X_test)

    print(loaded_model_preds)


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super(LinearRegressionModelV2, self).__init__()
        self.linear_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def whole():
    # Create device agnostic code
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Create data using linear regression formula
    # y = weight * X + bias
    # y = bx + a
    weight = 0.7
    bias = 0.3

    # Create range values
    start = 0
    end = 1
    step = 0.02

    # Create X and y (features and labels)
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # split data
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # plot the data
    plot_predictions(X_train, y_train, X_test, y_test)

    # Build the model
    torch.manual_seed(42)
    model_1 = LinearRegressionModelV2()
    print(model_1, model_1.state_dict())

    # Set model to use device
    model_1.to(device)

    ## Train model
    # Loss function
    # Optimizer
    # Training loop
    # Testing loop

    loss_fn = nn.L1Loss()  # Same as MAE
    optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

    # Training loop
    torch.manual_seed(42)
    epochs = 200
    epoch_count = []
    loss_values = []
    test_loss_values = []

    # Put data on target device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(epochs):
        model_1.train()

        # Forward pass
        y_pred = model_1(X_train)

        # Calculate loss
        loss = loss_fn(y_pred, y_train)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Perform back propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        ## Testing
        model_1.eval()
        with torch.inference_mode():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Test Loss: {test_loss}")

    print(model_1.state_dict())

    with torch.inference_mode():
        y_preds = model_1(X_test)
    print(y_preds)
    X_train = X_train.cpu()
    y_train = y_train.cpu()
    X_test = X_test.cpu()
    y_test = y_test.cpu()
    y_preds = y_preds.cpu()

    plot_predictions(
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        predictions=y_preds,
    )

    torch.save(model_1.state_dict(), f"models/01_pytorch_workflow_model_1.pth")


def load_whole():
    torch.manual_seed(42)

    weight = 0.7
    bias = 0.3

    # Create range values
    start = 0
    end = 1
    step = 0.02

    # Create X and y (features and labels)
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # split data
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    model_1_loaded = LinearRegressionModelV2()
    model_1_loaded.load_state_dict(torch.load("models/01_pytorch_workflow_model_1.pth"))

    model_1_loaded.eval()
    with torch.inference_mode():
        y_preds = model_1_loaded(X_test)

    plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)


if __name__ == "__main__":
    # main()
    # test_model()
    # load_model()
    # whole()
    load_whole()
