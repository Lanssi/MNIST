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
        #Original shape: (1, 1, 28, 28) for (batch, channel, height, width)
        #Maxpool before ReLU so that we can accelerate the computation
        out_chan=3
        self.conv1 = nn.Conv2d(1, out_chan, 3, 2, 1) #(14, 14)
        self.conv2 = nn.Conv2d(1, out_chan, 5, 3, 2) #(10, 10)
        self.conv3 = nn.Conv2d(1, out_chan, 7, 4, 3) #(7, 7)
        self.fc1 = nn.Linear(out_chan*(14*14+10*10+7*7), 500)
        self.fc2 = nn.Linear(500, 100)
        self.head = nn.Linear(100, 10)
        

    def forward(self, x):
        #Convolution
        path1 = F.relu(self.conv1(x)).flatten(1)
        path2 = F.relu(self.conv2(x)).flatten(1)
        path3 = F.relu(self.conv3(x)).flatten(1)
        output = torch.cat((path1, path2, path3), 1)
        #Fully connected
        output = F.relu(self.fc1(output)) 
        output = F.relu(self.fc2(output)) 
        output = self.head(output)
        
        return output
    
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
