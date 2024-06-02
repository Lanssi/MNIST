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
        self.EPOCHS = 20
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
        #Original shape: (1, 1, 28, 28) for (batch, channel, height, width)
        #Maxpool before ReLU so that we can accelerate the computation
        cs = 30
        self.model = nn.Sequential(
                nn.Conv2d(1, cs, 9, groups=1), #(20, 20)
                nn.ReLU(),
                nn.Conv2d(cs, cs, 20, groups=cs), #(1, 1)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(cs, 10, True),
        )

    def forward(self, x):
        return self.model(x)
    
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
