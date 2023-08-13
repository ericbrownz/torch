import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('lenet_cifar10.pth'))

    im = Image.open('fig/dog.jpeg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        print(outputs)
        print(torch.max(outputs, 1))
        _, predict = torch.max(outputs, 1)
    print(classes[int(predict.numpy())])


if __name__ == '__main__':
    main()
