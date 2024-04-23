""" FC Nueral Network for MNIST"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Configuration"""
class Config():
    def __init__(self, device):
        # Model hyperparameter
        self.input_size = 28*28
        self.hidden_size = 100
        self.output_size = 10

        # Training hyperparameter
        self.DEVICE = device
        self.EPOCHS = 10
        self.LEARNING_RATE = 0.05
        self.MOMENTUM = 0.5


"""Model definition"""
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train(self, train_loader, test_loader):
        # Put model to specified device
        self.to(self.config.DEVICE)

        # Define loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.LEARNING_RATE, 
                momentum=self.config.MOMENTUM)

        for epoch in range(self.config.EPOCHS):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, self.config.input_size).to(self.config.DEVICE)  #-1 means automatically deduced
                labels = labels.to(self.config.DEVICE)

                outputs = self.forward(images)  #output shape: (batch_size, output_size)
                loss = loss_func(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx+1) % 100 == 0:
                    print('Train: epoch {} [{}/{} ({:3.0f}%)], Loss {:.6f}'.format(
                        epoch+1, (batch_idx+1)*len(images), len(train_loader.dataset), 
                        100.*(batch_idx+1)*len(images)/len(train_loader.dataset), loss.item()))

            self.test(test_loader)
            print('')

    def test(self, test_loader):
        # Put model to specified device
        self.to(self.config.DEVICE)

        correct=0

        with torch.no_grad():
            for (images, labels) in test_loader:
                images = images.reshape(-1, self.config.input_size).to(self.config.DEVICE)  #-1 means automatically deduced
                labels = labels.to(self.config.DEVICE)

                outputs = self.forward(images)  #output shape: (batch_size, output_size)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        print("Test: accuracy: {}/{} ({:.2f}%)".format(
            correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
