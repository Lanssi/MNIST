import torch
import torchvision
import matplotlib.pyplot as plt


"""Configuration"""
DOWNLOAD_MNIST = False
MODEL_TYPE = "FC"   #MODEL_TYPE can be either FC or CNN


"""Hyperparameter"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
LEARNING_RATE = 0.05
MOMENTUM = 0.5
EPOCHS = 5


"""Load datasets"""
data_path = "~/MNIST/data"
train_set = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
)

test_set = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
)


"""Print the data"""
"""
nrows=2
ncols=3

for i in range(nrows * ncols):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(train_set.train_data[i].numpy(), cmap="gray")
    plt.title("Ground Truth: {}".format(train_set.train_labels[i]))

plt.show()
"""


"""Get dataloader"""
train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
)

test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=True
)


"""Get model

Note that configuration/hyperparameter is defined in the corresponding model file
e.g. in FC.py or CNN.py
"""
"""
if MODEL_TYPE == "FC":
    import FC
    config = FC.Config()
    model = FC.Model()
#TODO: Implement CNN
"""
import FC
model = FC.Model(FC.Config()).to(DEVICE)


"""Define loss function and optimizer"""
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


"""Train function"""
def train():
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28*28).to(DEVICE)  #-1 means automatically deduced
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 100 == 0:
                print("Epoch: [{}/{}], Step [{}/{}], Loss: {:.4f}"
                        .format(epoch, EPOCHS, batch_idx+1, total_step, loss.item()))
    

"""Test function"""
def test():
    correct=0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(DEVICE)   #-1 means automatically deduced
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _,  predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print("Test accuracy: {}/{} ({:.0f})%"
            .format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


train()
test()
