import torch.nn.functional as F
import torch


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


def hounsfield_clip(image):
    """
    Clip according to the useful Hounsfield depth
    Info taken from https://en.wikipedia.org/wiki/Hounsfield_scale
    Params:
        image: input image (tensor or numpy.array)
    Returns:
        image clipped according to Hounsfield scale
    """
    min_value = -60 # lower bound for fat tissue
    max_value = 180 # upper bound of the tumour
    return torch.clip(image, min_value, max_value)


def normalize(image):
    """
    Normalize the image to with min-max normalization
    Params:
        image: input image (tensor or numpy.array)
    Returns:
        normalized image with values between 0 and 1
    """
    return (image - image.min()) / (image.max() - image.min())


def torch_equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""

    def scale_channel(im):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :] * 1023
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=1024, min=0, max=1023)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 1023

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 1023)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im) / 1023

        return result

    return scale_channel(image)
