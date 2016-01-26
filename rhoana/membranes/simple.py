def intensity_as_probability(image, invert=False):
    '''Convert input image to membrane probabilities, linearly.

    Input is intensity is mapped linearly to membrane probabilities,
    with [min, max] = [0, 1] (or [0, 1] if invert is True).
    '''

    lo = image.min()
    hi = image.max()
    if invert:
        return (hi - image) / (hi - lo)
    return (image - lo) / (hi - lo)
