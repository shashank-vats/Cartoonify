import scipy.ndimage as sc
import matplotlib as mpt
import matplotlib.pyplot as plt
import numpy as np


mpt.use('TkAgg')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def threshold_filter(fig, thr):
    fig_copy = fig.copy()
    for row in range(fig.shape[0]):
        for col in range(fig.shape[1]):
            if fig[row][col] > thr:
                fig_copy[row][col] = 255
            else:
                fig_copy[row][col] = round(fig[row][col]**(1/1.21))
    # show_images([fig_copy], titles=['Inside the func'])
    return fig_copy


img = plt.imread('./Shnk1.jpeg')
img = rgb2gray(img)

img_low_pass = sc.gaussian_filter(input=img, sigma=5)
depth = 4

# img_high_pass = (1 + depth)*img - (depth - 0.3)*img_low_pass
img_high_pass = img_low_pass.copy()
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        img_high_pass[row][col] = min(255, max(0, round((1 + depth) *
                                                        img[row][col] - (depth - 0.3)*img_low_pass[row][col])))

improved_img = threshold_filter(img_high_pass, 128)
# print(type(img))
# print(img.shape)
# print(improved_img)
show_images([img, img_low_pass, img_high_pass, improved_img + img],
            titles=['original image', 'low pass filtered', 'high pass filtered', 'improved image'])
