'''
Train an Alexnet on Mnist dataset.
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from models.Alexnet import AlexNet

lr = 0.001
epochs = 20
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()


model = AlexNet()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Start Training
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, epochs, i+1,
                     len(train_dataset)//batch_size,
                     running_loss/100))
            running_loss = 0.0

    # Test model on test_dataloader after an epoch.
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    # Save checkpoint after each epoch.
    acc = 100 * correct / total
    torch.save(model.state_dict(), f"ckpt/Alexnet_mnist_acc={acc}.pth")


# Training log
# Epoch [2/20], Iter [920/937] Loss: 0.0150
# Epoch [2/20], Iter [930/937] Loss: 0.0184
# Accuracy of the network on the test images: 94 %
# Epoch [3/20], Iter [920/937] Loss: 0.0099
# Epoch [3/20], Iter [930/937] Loss: 0.0139
# Accuracy of the network on the test images: 96 %
