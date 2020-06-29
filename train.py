import os

import torch
from torch.optim.lr_scheduler import StepLR

from utils import calculate_correct_predictions


def train(model, optimizer, loss_fn, epochs, train_loader, device, model_chckpt_path, checkpoint_save_interval,
          model_path, load_chckpt, log_interval):
    epoch_start = 0

    scheduler = StepLR(optimizer, 30, 0.1)

    if load_chckpt and os.path.isfile(model_chckpt_path):
        checkpoint = torch.load(model_chckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        print("Training checkpoints found. Starting training from epoch %d." % epoch_start)

    model.train()
    for epoch in range(epoch_start, epochs):
        running_loss = 0.0
        processed_items = 0
        correct_predictions = 0
        for batch_num, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            optimizer.zero_grad()
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()

            _, correct = calculate_correct_predictions(targets, out)
            running_loss += loss.item()
            processed_items += out.size()[0]
            correct_predictions += correct

            if (batch_num + 1) % log_interval == 0:
                print('[Epoch %d, Batch %4d] Loss: %.10f, Accuracy: %.5f' %
                      (epoch + 1, batch_num + 1, running_loss / processed_items, correct_predictions / processed_items))

        if epoch % checkpoint_save_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, model_chckpt_path)
    torch.save(model.state_dict(), model_path)
