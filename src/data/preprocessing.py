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
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode=mode)[0]