import os
from datetime import datetime
from collections import OrderedDict

import click
import torch
import progressbar
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets as dset

from resnet.resnet import ResNet, Bottleneck


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')


def train(epoch, model, loader, criterion, optimizer, use_cuda=False,
          tracking=None):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    total = 0
    total_correct = 0
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
        batch_size = targets.size(0)
        correct = predictions.eq(targets.data).cpu().sum()
        total += batch_size
        total_correct += correct

        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['epoch'] = epoch
            result['batch'] = batch_index
            result['batch_size'] = batch_size
            result['ncorrect'] = correct
            result['loss'] = loss.data[0]
            save_result(result, tracking)

        bar.update(batch_index)

    accuracy = (1. * total_correct) / total
    print('Training accuracy of {}'.format(accuracy))
    return accuracy


def test(epoch, model, loader, use_cuda=False, tracking=None):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    total = 0
    total_correct = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = Variable(inputs)
        targets = Variable(targets)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        _, predictions = torch.max(outputs.data, 1)
        batch_size = targets.size(0)
        correct = predictions.eq(targets.data).cpu().sum()
        total += batch_size
        total_correct += correct

        if tracking is not None:
            result = OrderedDict()
            result['timestamp'] = datetime.now()
            result['epoch'] = epoch
            result['batch'] = batch_index
            result['batch_size'] = batch_size
            result['ncorrect'] = correct
            save_result(result, tracking)

        bar.update(batch_index)

    accuracy = (1. * total_correct) / total
    print('Test accuracy of {}'.format(accuracy))
    return accuracy


def save(model, directory, epoch, accuracy, use_cuda=False, filename=None):
    state = {
        'model': model.module if use_cuda else model,
        'epoch': epoch,
        'accuracy': accuracy
    }

    filename = filename or 'checkpoint_{}.t7'.format(epoch)
    torch.save(state, os.path.join(directory, filename))


def load(path):
    assert os.path.exists(path)
    state = torch.load(path)
    model = state['model']
    epoch = state['epoch']
    accuracy = state['accuracy']
    return model, epoch, accuracy


@click.command()
@click.option('--dataset-dir', default='./data/cifar10')
@click.option('--checkpoint-dir', '-c', default='./checkpoints')
@click.option('--restore', '-r')
@click.option('--tracking', '-t', is_flag=True)
@click.option('--cuda', is_flag=True)
@click.option('--epochs', '-e', default=200)
@click.option('--batch-size', '-b', default=32)
@click.option('--learning-rate', '-l', default=1e-3)
def main(dataset_dir, checkpoint_dir, restore, tracking, cuda, epochs,
         batch_size, learning_rate):
    checkpoint_dir = os.path.join(
        checkpoint_dir, "{:.0f}".format(datetime.utcnow().timestamp()))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if tracking:
        train_results_file = os.path.join(checkpoint_dir, 'train_results.csv')
        test_results_file = os.path.join(checkpoint_dir, 'test_results.csv')
    else:
        train_results_file = None
        test_results_file = None

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

    if restore is not None:
        print('Restoring model')
        model, start_epoch, best_accuracy = load(restore)
        start_epoch += 1
        print('Starting accuracy is {}'.format(best_accuracy))
    else:
        print('Building model')
        best_accuracy = -1
        start_epoch = 1
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)

    if use_cuda:
        print('Copying model to GPU')
        model.cuda()
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
    criterion = nn.CrossEntropyLoss()

    # Other parameters?
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, start_epoch + epochs):
        print('Epoch {} of {}'.format(epoch, start_epoch + epochs - 1))
        train(epoch, model, train_loader, criterion, optimizer,
              use_cuda=use_cuda, tracking=train_results_file)
        accuracy = test(epoch, model, test_loader, use_cuda=use_cuda,
                        tracking=test_results_file)

        if accuracy > best_accuracy:
            print('New best model!')
            save(model, checkpoint_dir, epoch, accuracy, use_cuda=use_cuda,
                 filename='checkpoint_best_model.t7')
        save(model, checkpoint_dir, epoch, accuracy, use_cuda=use_cuda)

    print('Finished training')
    test(model, test_loader, use_cuda=use_cuda)


if __name__ == '__main__':
    main()
