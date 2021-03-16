import torch.nn.functional as F


def resize(image, size=512, mode="bilinear"):
    """
    resize the 3D image to the shape we want, in width and length
    Params:
        image: input image (tensor or numpy.array)
        size: shape of the output image
    Returns:
        resized image in width and length
    """
    if size == 512:
        return image
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode=mode)[0]


def normalize(image):
    """
    Normalize the image to with min-max normalization
    Params:
        image: input image (tensor or numpy.array)
    Returns:
        normalized image with values between 0 and 1
    """
    return (image - image.min()) / (image.max() - image.min())
