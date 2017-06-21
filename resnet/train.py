import click
import torch
import progressbar
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets as dset

from resnet.resnet import ResNet, Bottleneck


def train(model, loader, criterion, optimizer, use_cuda=False):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    total = 0
    correct = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = Variable(inputs)
        targets = Variable(targets)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predictions.eq(targets.data).cpu().sum()
        print('Training accuracy of {}'.format((1.*correct) / total))

        bar.update(batch_index)
    print('Training accuracy of {}'.format((1.*correct) / total))


def test(model, loader, use_cuda=False):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    total = 0
    correct = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = Variable(inputs)
        targets = Variable(targets)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        _, predictions = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predictions.eq(targets.data).cpu().sum()

        bar.update(batch_index)

    print('Test accuracy of {}'.format((1.*correct) / total))


@click.command()
@click.option('--dataset-dir', default='./data/cifar10')
@click.option('--checkpoint-dir', '-c', default='./checkpoints')
@click.option('--cuda', is_flag=True)
@click.option('--epochs', '-e', default=200)
@click.option('--batch-size', '-b', default=32)
@click.option('--learning-rate', '-l', default=1e-3)
def main(dataset_dir, checkpoint_dir, cuda, epochs, batch_size, learning_rate):
    print("Preparing data:")
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = dset.CIFAR10(root=dataset_dir, train=True, download=True,
                            transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = dset.CIFAR10(root=dataset_dir, train=False, download=True,
                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False)

    use_cuda = cuda or torch.cuda.is_available()

    print('Building model')
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)

    if use_cuda:
        print('Copying model to GPU')
        model.cuda()
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss()

    # Other parameters?
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        print('Epoch {} of {}'.format(epoch, epochs))
        train(model, train_loader, criterion, optimizer, use_cuda=use_cuda)
        test(model, test_loader, use_cuda=use_cuda)


if __name__ == '__main__':
    main()
