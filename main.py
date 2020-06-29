import argparse
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms

from RDenseCnn import RDenseCNN
from train import train
from test import test


def load_data(dataset_name, batch_size):
    if dataset_name == "fmnist":
        trans = transforms.Compose([
            transforms.RandomAffine(180, (0.125, 0.125)),
            transforms.Resize(32),
            transforms.ToTensor()])
        fashion_mnist_train = datasets.FashionMNIST('./datasets', train=True, download=True, transform=trans)
        fashion_mnist_test = datasets.FashionMNIST('./datasets', train=False, download=True, transform=trans)
        train_loader = torch.utils.data.DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)
        num_classes = 10
        num_channels = 1
        num_rd_blocks = 2
        labels = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
        return train_loader, test_loader, num_channels, num_rd_blocks, num_classes, labels

    # todo add for other datasets


def parse_arguments():
    parser = argparse.ArgumentParser(description="RDenseCNN implementation")

    # Generic arguments
    parser.add_argument("--batch-size", type=int, default=200, help="batch size for training (default: 256)")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train for (default: 30)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=0.0004,
                        help="Weight decay for Adam optimizer (default: 0.004)")
    parser.add_argument("--checkpoint-save-interval", type=int, default=10,
                        help="Interval at which checkpoints are created")
    parser.add_argument("--use-gpu", type=int, default=1,
                        help="flag (0/1) to indicate if gpu should be used (default: 1)")
    parser.add_argument(
        "--log-interval", type=int, default=20, help="number of batches after which to log loss (default: 20)")
    parser.add_argument("--model-path", type=str, default="./model_data/model.weights",
                        help="path to model weights. After training weights will be saved there and if already present,"
                             "training is skipped")
    parser.add_argument("--model-checkpoint-path", type=str, default="./model_data/model.weights.chckpt",
                        help="path to model checkpoint file in case that checkpoints during training should be created")

    # RDenseCNN-specific arguments
    parser.add_argument("--k", type=int, default=12, help="Growth rate in one residual dense block (default: 12)")
    parser.add_argument("--m", type=int, default=16,
                        help="Number of dense layers per residual dense block (default: 16)")
    parser.add_argument("--dataset", type=str, default="fmnist",
                        help="dataset to use can be fmnist, mnist, cifar-10 or cifar-100")
    return parser.parse_args()


def main():
    args = parse_arguments()

    train_loader, test_loader, num_channels, num_rd_blocks, num_classes, labels = \
        load_data(args.dataset, args.batch_size)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    model = RDenseCNN(num_channels, num_rd_blocks, args.m, args.k, num_classes)
    model = model.to(device)
    print(model.num_parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    weights_saved = os.path.isfile(args.model_path)
    chckpt_present = os.path.isfile(args.model_checkpoint_path)
    if not weights_saved:
        console.print("[bold blue] (RDenseCNN) [bold blue] No saved weights found. Starting model training!")
        train(model, optimizer, loss_fn, args.epochs, train_loader, device,
              args.model_checkpoint_path, args.checkpoint_save_interval,
              args.model_path, chckpt_present, args.log_interval)

    if weights_saved:
        model.load_state_dict(torch.load(args.model_path))
        test(model, test_loader, device, num_classes, labels)


if __name__ == "__main__":
    main()
