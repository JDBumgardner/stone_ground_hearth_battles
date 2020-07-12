# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

torch.manual_seed(1)


class HearthstoneNet(nn.Module):
    def __init__(self):
        super(HearthstoneNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)

        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x)

        return x

def main():
    tensor1 = torch.rand((64,))
    print(tensor1)

    num_epochs = 10
    batch_size = 32

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = HearthstoneNet()
    device = torch.device('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    model.to(device)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


if __name__ == '__main__':
    main()