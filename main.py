import cv2
import numpy as np


def energy_function(image):
    """
    Compute the magnitude gradient of the image.

    :param image: Numpy array gray image
    :return: Numpy array image
    """

    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    return (mag / mag.max() * 255).astype(np.uint8)


def find_seam(image):
    """
    Compute the lowest energy seam using dynamic programming.

    :param image: Numpy array gray image
    :return: List of pixel tuples
    """

    height, width = image.shape
    dp = np.zeros(image.shape)
    dp[0] = image[0]
    for r in xrange(1, height):
        for c in xrange(width):
            if c == 0:
                dp[r, c] = min(dp[r - 1, c], dp[r - 1, c + 1])
            elif c == width - 1:
                dp[r, c] = min(dp[r - 1, c], dp[r - 1, c - 1])
            else:
                dp[r, c] = min(dp[r - 1, c + 1], dp[r - 1, c], dp[r - 1, c - 1])
            dp[r, c] += image[r, c]

    min_val = float("INF")
    min_pointer = None
    for c in range(width):
        if dp[height - 1][c] < min_val:
            min_val = dp[height - 1][c]
            min_pointer = c

    path = []
    pos = (height - 1, min_pointer)
    path.append(pos)
    while pos[0] != 0:
        value = dp[pos] - image[pos]
        r, c = pos
        if c == 0:
            if value == dp[r - 1, c + 1]:
                pos = (r - 1, c + 1)
            else:
                pos = (r - 1, c)
        elif c == width - 1:
            if value == dp[r - 1, c - 1]:
                pos = (r - 1, c - 1)
            else:
                pos = (r - 1, c)
        else:
            if value == dp[r - 1, c + 1]:
                pos = (r - 1, c + 1)
            elif value == dp[r - 1, c]:
                pos = (r - 1, c)
            else:
                pos = (r - 1, c - 1)
        path.append(pos)
    return path


def remove_seam(image, image_orientation='v'):
    """
    Converts RGB image to gray image and remove seam from original image. Rotates according to orientation if necessary.

    :param image: Numpy array RGB image
    :param image_orientation: string 
    :return: Numpy array image
    """

    if image_orientation == 'h':
        image = np.rot90(image, k=1)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width, channels = image.shape
    size = (height, width - 1, channels)

    seam = [r * (width * channels) + c * channels + rgb
            for r, c in find_seam(energy_function(gray_image)) for rgb in range(channels)]
    image = np.reshape(np.delete(image.ravel(), seam), size)

    if image_orientation == 'h':
        return np.rot90(image, k=3)
    return image


if __name__ == "__main__":
    img = cv2.imread("tower.jpg", 1)
    for i in range(200):
        print i
        orientation = 'v' if i % 5 else 'h'
        img = remove_seam(img, orientation)
        cv2.imshow("image", img)
        cv2.waitKey(0)
