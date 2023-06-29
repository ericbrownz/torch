import torch
import torchvision
import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
# import numpy as np

from model import LeNet
import torch.nn as nn
import torch.optim as optim


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # (input-0.5)*0.5

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5000,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # [C, H, W] to [H, W, C]
        plt.show()
        plt.savefig('fig/test0_offical_demo.png')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    # show images
    imshow(torchvision.utils.make_grid(images))
    """

    # get some random test images
    test_dataiter = iter(testloader)
    test_images, test_labels = next(test_dataiter)

    net = LeNet()
    criterion = nn.CrossEntropyLoss()   # loss func
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # [5,  1000] loss: 0.258, accuracy: 0.616
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # [5,  1000] loss: 0.362, accuracy: 0.496

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(test_images)  # [batch, 10] find the second dim
                    _, predict = torch.max(outputs, 1)    # [val, idx] choose dim 1
                    accuracy = (predict == test_labels).sum().item() / test_labels.size(0)

                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, accuracy: {accuracy:.3f}')
                    running_loss = 0.0

    print('Finished Training')

    PATH = './lenet_cifar10.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
