import torch
import torchvision
import matplotlib.pyplot as plt


"""Configuration"""
DOWNLOAD_MNIST = False
MODEL_TYPE = "CNN"   #MODEL_TYPE can be either FC or CNN
DEVICE = 'cuda'
if DEVICE == 'cpu':
    pass
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""Hyperparameter
Note that hyperparameters of model are defined in the corresponding model file,
e.g. in FC.py or CNN.py
"""
BATCH_SIZE = 100


"""Load datasets"""
data_path = "~/Project/MNIST/data"
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


"""Get model"""
if MODEL_TYPE == 'FC':
    import FC
    model = FC.Model(FC.Config(DEVICE))
if MODEL_TYPE == 'CNN':
    import CNN
    model = CNN.Model(CNN.Config(DEVICE))
if MODEL_TYPE == 'CNN2':
    import CNN2
    model = CNN2.Model(CNN2.Config(DEVICE))

model.test(test_loader)
model.train(train_loader, test_loader)
