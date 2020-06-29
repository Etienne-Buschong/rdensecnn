import matplotlib.pyplot as plt
import numpy as np
import torch


def im_show(imgs, predictions, labels, num_imgs_to_show, num_classes):
    plt.figure(figsize=(10, 10))
    for i in range(num_imgs_to_show):
        plt.subplot(2 * num_imgs_to_show ** .5, 2 * num_imgs_to_show ** .5, 2 * i + 1)
        plt.xticks([])
        plt.yticks([])
        np_img = imgs[i].squeeze().numpy()
        plt.imshow(np_img, cmap=plt.cm.binary)
        plt.xlabel(labels[predictions[i].argmax()])

        plt.subplot(2 * num_imgs_to_show ** .5, 2 * num_imgs_to_show ** .5, 2 * i + 2)
        plt.xticks(np.arange(num_classes))
        plt.yticks([])
        plt.bar(np.arange(num_classes), predictions[i].numpy())
    plt.show()


def calculate_correct_predictions(correct_labels, network_output):
    _, predictions = torch.max(network_output, 1)
    correct_vector = (predictions == correct_labels).cpu().numpy()
    return correct_vector, correct_vector.sum()

