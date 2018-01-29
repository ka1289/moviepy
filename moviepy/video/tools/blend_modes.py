import numpy as np
import decorator
import time


@decorator.decorator
def make_rgba_image(f, image, layer, opacity):
    '''
    Using this decorator to manipulate frames to have 4 channels
    '''
    # opacity = 1
    # print('opacity', opacity)
    image_org = image.copy()
    layer_org = layer.copy()
    if image.shape[2] < 4:
        rows_, columns_, depth = image_org.shape[:3]
        image = np.zeros((rows_, columns_, 4), np.float)
        image[:, :, :3] = image_org
        image[:, :, 3] = 255
    if layer.shape[2] < 4:
        rows_, columns_, depth = layer_org.shape[:3]
        layer = np.zeros((rows_, columns_, 4), np.float)
        layer[:, :, :3] = layer_org
        layer[:, :, 3] = 255
    if opacity > 1:
        opacity /= 255

    return f(image, layer, opacity)[:, :, :3]


@make_rgba_image
def overlay(img_in, img_layer, opacity):
    """
    Apply overlay blend effect on input image
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:,:,:3] * (img_in[:,:,:3] + (2 * img_layer[:,:,:3]) * (1 - img_in[:,:,:3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0

@make_rgba_image
def burn(img_in, img_layer, opacity):
    """
    Apply burn blend effect on input image
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0
    
    ratio = _compose_alpha(img_in, img_layer, opacity)
    
    comp = 1 - (1-img_in[:,:,:3])/img_layer[:,:,:3]

    # comp[comp <0] = 0 

    # comp[comp >1] = 1
    comp = np.clip(comp,0,1)
    
    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans

    return img_out * 255.0



@make_rgba_image
def soft_light(img_in, img_layer, opacity):
    """
    Apply soft light blending mode of a layer on an image.
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    # The following code does this:
    #   multiply = img_in[:, :, :3]*img_layer[:, :, :3]
    #   screen = 1.0 - (1.0-img_in[:, :, :3])*(1.0-img_layer[:, :, :3])
    #   comp = (1.0 - img_in[:, :, :3]) * multiply + img_in[:, :, :3] * screen
    #   ratio_rs = np.reshape(np.repeat(ratio,3),comp.shape)
    #   img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)

    comp = (1.0 - img_in[:, :, :3]) * img_in[:, :, :3] * img_layer[:, :, :3] \
        + img_in[:, :, :3] * (1.0 - (1.0 - img_in[:, :, :3]) * (1.0 - img_layer[:, :, :3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def lighten_only(img_in, img_layer, opacity):
    """
    Apply lighten only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Lighten_Only>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = lighten_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.maximum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def screen(img_in, img_layer, opacity):
    """
    Apply screen blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Screen>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = screen(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = 1.0 - (1.0 - img_in[:, :, :3]) * (1.0 - img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def dodge(img_in, img_layer, opacity):
    """
    Apply dodge blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Dodge_and_burn>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = dodge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3] / (1.0 - img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def addition(img_in, img_layer, opacity):
    """
    Apply addition blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Addition>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = addition(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    # ratio = _compose_alpha(img_in, img_layer, opacity)
    ratio_val = _compose_alpha_addition(img_in[:, :, 3][0][0], img_layer[:, :, 3][0][0], opacity)
    ratio_rs = np.full((img_in.shape[0], img_in.shape[1], 3), ratio_val)

    comp = img_in[:, :, :3] + img_layer[:, :, :3]

    # ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    # ratio_rs = np.dstack((ratio, ratio, ratio))
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    # img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    # img_out = np.dstack((img_out, img_in[:, :, 3]))
    # img_out[img_out == np.nan] = 0
    # img_out[img_out == np.inf] = 1
    return img_out * 255.0


@make_rgba_image
def darken_only(img_in, img_layer, opacity):
    """
    Apply darken only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Darken_Only>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = darken_only(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def multiply(img_in, img_layer, opacity):
    """
    Apply multiply blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Multiply>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = multiply(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def hard_light(img_in, img_layer, opacity):
    """
    Apply hard light blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Hard_Light>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = hard_light(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.greater(img_layer[:, :, :3], 0.5) * np.minimum(1.0 - ((1.0 - img_in[:, :, :3])
                                                                    * (1.0 - (img_layer[:, :, :3] - 0.5) * 2.0)), 1.0) \
        + np.logical_not(np.greater(img_layer[:, :, :3], 0.5)) * np.minimum(img_in[:, :, :3]
                                                                            * (img_layer[:, :, :3] * 2.0), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def difference(img_in, img_layer, opacity):
    """
    Apply difference blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = difference(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]
    comp[comp < 0.0] *= -1.0

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def subtract(img_in, img_layer, opacity):
    """
    Apply subtract blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Subtract>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = subtract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def grain_extract(img_in, img_layer, opacity):
    """
    Apply grain extract blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Extract>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_extract(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] - img_layer[:, :, :3] + 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def grain_merge(img_in, img_layer, opacity):
    """
    Apply grain merge blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Merge>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_merge(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] + img_layer[:, :, :3] - 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


@make_rgba_image
def divide(img_in, img_layer, opacity):
    """
    Apply divide blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Divide>`__.

    Example::

        import img_filters_c, cv2
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = divide(img_in,img_layer,0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype == np.float, 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype == np.float, 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum((256.0 / 255.0 * img_in[:, :, :3]) / (1.0 / 255.0 + img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def _compose_alpha(img_in, img_layer, opacity):
    """
    Calculate alpha composition ratio between two images.
    """

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio


def _compose_alpha_addition(img_in_mask, img_layer_mask, opacity):
    """
    Calculate alpha composition ratio between two images.
    """

    comp_alpha = np.minimum(img_in_mask, img_layer_mask) * opacity
    new_alpha = img_in_mask + (1.0 - img_in_mask) * comp_alpha
    ratio = comp_alpha / new_alpha
    return ratio
