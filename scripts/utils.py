def img_thresholding(img, threshold):
    img[img > threshold] = 255
    img[img < threshold] = 0
    return img
