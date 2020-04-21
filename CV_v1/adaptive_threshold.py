import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans


def adaptive_threshold(median_blurred_image):
    img = cv.medianBlur(median_blurred_image, 5)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return th2


def otsu_threshold(gaussian_blurred_image):
    ret, th3 = cv.threshold(gaussian_blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th3


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
            :param titles: names to show as titles of images
            :param images: list of images to show
            :param cols: Numbers od cols to show
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


def color_quantize(image, num_bits):
    image = convert_cv_to_pil(image)
    return convert_pil_to_cv(ImageOps.posterize(image, num_bits))


def color_quantize_alternate(image, k):
    # Grab the height and width of image
    (h, w) = image.shape[:2]

    # Convert the image from the RGB color space to L*a*b color space
    # -- since we will be clustering using k-means which is based on
    # the euclidean distance, we'll use the L*a*b color space where the
    # euclidean distance implies perceptual meaning
    image = cv.cvtColor(image, cv.COLOR_RGB2LAB)

    # reshape the image in a feature vector so that k-means can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b to RGB
    return cv.cvtColor(quant, cv.COLOR_LAB2RGB)


def convert_pil_to_cv(pil_image):
    return np.array(pil_image)


def convert_cv_to_pil(cv_image):
    return Image.fromarray(cv_image)


def make_composite_image(edge_enhanced_image, color_quantized_image, otsu_image):
    color_quantized_image = convert_pil_to_cv(color_quantized_image)
    result = cv.addWeighted(color_quantized_image, 0.7, edge_enhanced_image, 0.3, 0)

    # Add a little bit of shading too
    return cv.addWeighted(result, 0.8, otsu_image, 0.2, 0)


def convert_grayscale_to_rgb(gray):
    return cv.cvtColor(gray, cv.COLOR_GRAY2RGB)


if __name__ == '__main__':
    image_path = 'Shnk1.jpeg'
    grayscale_image = cv.imread(image_path, 0)
    adaptive_result = adaptive_threshold(cv.medianBlur(grayscale_image, 5))
    otsu_result = otsu_threshold(cv.GaussianBlur(grayscale_image, (5, 5), 0))
    gaussian_filtered_adaptive_result = cv.GaussianBlur(adaptive_result, (5, 5), 0)
    color_image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    color_quantization_result = color_quantize_alternate(color_image, 64)
    color_quantized_edge_enhanced_result = convert_grayscale_to_rgb(color_quantize(gaussian_filtered_adaptive_result, 2))
    # print('shape of edge enhanced image: ', color_quantized_edge_enhanced_result.shape)
    final_result = make_composite_image(color_quantized_edge_enhanced_result,
                                        color_quantization_result,
                                        convert_grayscale_to_rgb(otsu_result))
    image_list = [color_image, final_result]
    titles = ['Original', 'Final Result']
    show_images(images=image_list, titles=titles)
