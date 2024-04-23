""" CNN Nueral Network for MNIST """
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Configuration """
class Config():
    def __init__(self, device):
        #Model hyperparameter
          #Model paramater is too much, so avoid definition here

        #Training hyperparameter
        self.DEVICE = device
        self.EPOCHS = 10
        self.LEARNING_RATE = 0.001
    
""" Model definition
Note that the formula is: out_shape = (in_shape - kernel_size + 2*padding)/stride + 1
If we want out_shape = in_shape and stride = 1, then padding = (kernel_size - 1)/2
"""
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #Note that you can define your own model here, be aware of the shape however
        #Original shape: (1, 1, 28, 28) for (batch, channel, weight, width)
        #Maxpool before ReLU so that we can accelerate the computation
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2, padding=1), #(24, 24)
                #nn.MaxPool2d(kernel_size = 2), #(12, 12)
                nn.ReLU() #(12, 12)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=3, padding=2), #(8, 8)
                #nn.MaxPool2d(kernel_size = 2),
                nn.ReLU() #(8, 8)
        )
        self.fc1 = nn.Linear(in_features=20*5*5, out_features=250, bias=True)
        self.fc2 = nn.Linear(in_features=250, out_features=10, bias=True) 

    def forward(self, x):
        #Convolution
        x = self.conv1(x)
        x = self.conv2(x)
        #Fully connected
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def train(self, train_loader, test_loader):
        #Put model to specified device
        self.to(self.config.DEVICE)

        #Define loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        for epoch in range(self.config.EPOCHS):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)

                outputs = self.forward(images)
                loss = loss_func(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx+1)%100 == 0:
                    print('Train: epoch {} [{}/{} ({:3.0f}%)], loss {:.6f}'.format(
                        epoch+1, (batch_idx+1)*len(images), len(train_loader.dataset),
                        100.*(batch_idx+1)*len(images)/len(train_loader.dataset), loss.item()))

            self.test(test_loader)
            print('')

    def test(self, test_loader):
        #Put model to specified device
        self.to(self.config.DEVICE)
        
        correct = 0
        
        with torch.no_grad():
            for (images, labels) in test_loader:
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)

                outputs = self.forward(images) 
                _, predicted = torch.max(outputs.data, -1)
                correct += (predicted == labels).sum().item()

        print("Test: accuracy: {}/{} ({:.2f}%)".format(
            correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))
