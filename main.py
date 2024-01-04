import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine, ToTensor, Normalize
from PIL import Image
from mser import MSERProcessor

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        kaiming_uniform_(self.linear.weight)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_mser_images(image_path):
    mser_processor = MSERProcessor(image_path)
    mser_boxes = mser_processor.process_image()
    original_image = Image.open(image_path).convert('L')
    images = []
    for box in mser_boxes:
        roi = np.array(original_image)
        roi = roi[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        padded_image = np.pad(roi, [(10, 10), (10, 10)], 'constant', constant_values=255)
        cropped_image = Image.fromarray(padded_image)
        images.append(cropped_image)
    return images
    

def main():

    is_train = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if is_train:
        print(device)
        transform_train = transforms.Compose([
            RandomHorizontalFlip(p=0.3),
            RandomVerticalFlip(p=0.3),
            RandomRotation(degrees=(-90, 90)),
            RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomInvert(p=0.5),
            ToTensor(),
            Normalize((0.1307,), (0.3081,))

        ])

        transform_test = transforms.Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = MNIST('./data', train=True, download=True, transform=transform_train)
        test_dataset = MNIST('./data', train=False, transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

        model = ResNet18().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(1, 6):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

        torch.save(model.state_dict(), 'models/mnist_cnn.pt')
        print('Model saved to mnist_cnn2.pt')

    else:
        model = ResNet18()
        model.load_state_dict(torch.load('models/mnist_cnn.pt'))
        print('Model loaded from mnist_cnn.pt')

        mser_images = get_mser_images('digits.png')  # Get the list of images

        for mser_image in mser_images:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ])

            input_image = transform(mser_image)
            Image.fromarray(input_image.squeeze().numpy()).show()
            input_image.unsqueeze_(0)

            model.eval()
            with torch.no_grad():
                output = model(input_image)
                prediction = output.argmax(dim=1, keepdim=True).item()

            print(f'Predicted label: {prediction}')
            print('Output probabilities:', F.softmax(output, dim=1))

if __name__ == '__main__':
    main()
