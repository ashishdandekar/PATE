import numpy as np
import matplotlib.pyplot as plt

from struct import unpack

def loadmnist(imagefile, labelfile):
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Metadata for images
    images.read(4)      # skip the magic number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)
    y = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

def show_figures(images, labels):
    plt.figure(figsize=(20,4))
    for index, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap = plt.cm.gray)
        plt.title('Training: %i\n' % (label), fontsize = 20)

    plt.show()
