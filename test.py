import numpy as np
import torch

from utils import calculate_correct_predictions


def test(model, test_loader, device, num_classes, labels):
    correct = [0 for _ in range(num_classes)]
    total = [0 for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            _, predictions = torch.max(out, 1)
            real_batch_size = out.size()[0]
            comp, c = calculate_correct_predictions(targets, out)
            for i in range(real_batch_size):
                correct[targets[i]] += comp[i].item()
                total[targets[i]] += 1

    complete_correct = np.array(correct).sum()
    complete_total = np.array(total).sum()

    print('Accuracy of the network on %d test images: %.3f' % (complete_total, complete_correct / complete_total))
    for i in range(num_classes):
        print('Accuracy on class %s: %.3f' % (labels[i], correct[i] / total[i]))
