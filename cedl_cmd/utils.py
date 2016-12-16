"""
"""

import theano
from theano import tensor as T

import numpy
import h5py

import lasagne
from lasagne.layers.noise import DropoutLayer
from lasagne.layers import Layer, DenseLayer
from lasagne.layers import Conv2DLayer 
from lasagne.layers import MaxPool2DLayer
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast 
except:
    Conv2DLayerFast = Conv2DLayer
    MaxPool2DLayerFast = MaxPool2DLayer
from lasagne import nonlinearities, init
from lasagne.nonlinearities import rectify, sigmoid 


class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
    
    
def load_hdf(filename, idx_start=0, samples=10000, normalize=False, gradient=False, gamma=None,channel=None):
    print 'loading data',
    h = h5py.File(filename, 'r')
        
    if channel is None:
        X = h['galleries'][idx_start:idx_start+samples, ...]
    else: 
        X = h['galleries'][idx_start:idx_start+samples, channel, ...][:, None, ...]
        
    Y = h['labels'][idx_start:idx_start+samples, ...]
    if 'features' in h:
        Xfeat = h["features"][idx_start:idx_start+samples, ...]
    else:
        Xfeat = None
        
    X = X.astype(numpy.float32) / 255.
    
    if len(X.shape) == 3:
        X = X[:, None, :, :]
    elif len(X.shape) == 2:
        X = X[:, None, :]
        X = X.reshape(X.shape[0], 1, int(numpy.sqrt(X.shape[2])), int(numpy.sqrt(X.shape[2])))  
    
    if gradient:
        def grad_mag(arr):
            gx, gy = numpy.gradient(arr[0,...])
            mag = numpy.sqrt(gx**2 + gy**2)
            
            return mag[None, None, ...]    
        X = numpy.concatenate(map(grad_mag, X), axis=0)
    
    if normalize:
        min_ = numpy.percentile(X, normalize[0])
        max_ = numpy.percentile(X, normalize[1])
        
        X = (X-min_) / (max_ - min_)
        X = numpy.clip(X, 0, 1)
         
        if gamma is not None:
            X = X**gamma
    
          
    h.close()

    print 'done'
      
    print "***" * 5
    print "Range\t", X.min(), X.max()
    print "mean \t", X.mean()
    print "media\t", numpy.median(X)
    print "std  \t", X.std()
    
    return X, Xfeat, Y


def act(letter):
    if letter == "r":
        return rectify
    elif letter == "s":
        return sigmoid
    else:
        raise ValueError('non-linearity not understood')
    
    
def pprint_layers(lyr):
    if hasattr(lyr, "input_layer"):
        pprint_layers(lyr.input_layer)
            
    else:
        print "  Autoencoder Layout..."
        print "-"*80
        print "%26s\t%10s\t%12s\t%20s\t" % ("Layer type", "Name", "Dimensions", "Comment")
        
    aux = ""
    
    if isinstance(lyr, (Conv2DLayerFast, Conv2DLayer)):
        if isinstance(lyr.filter_size, int):
            fs = lyr.filter_size
        else:
            fs = lyr.filter_size[0] 
        aux = "%2d filters: %dx%d" % (lyr.num_filters, fs, fs)
    elif isinstance(lyr, (MaxPool2DLayer)):
        aux = "pool size: %dx%d" % lyr.pool_size
        #aux = "pool size: %dx%d" % (lyr.pool_size, lyr.pool_size)
    elif isinstance(lyr, (MaxPool2DLayerFast,)):
        aux = "pool size: %dx%d" % (lyr.pool_size, lyr.pool_size)
    elif isinstance(lyr, (DropoutLayer,)):
        aux = "drop prob.: %2.2f" % lyr.p
    elif isinstance(lyr, (DenseLayer,)):
        aux = "%d -> %d" % (numpy.prod(lyr.input_shape[1:]), lyr.num_units)
    
    if hasattr(lyr, "nonlinearity"):
        aux += " (%s)" % (lyr.nonlinearity.__name__)[0]
    aux += " "    
    print "%26s\t%10s\t%12s\t%20s\t" % (lyr.__class__.__name__, lyr.name, "x".join(map(str,lyr.output_shape[1:])), aux)
    
    
    
    
    
    
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.ones(out_shape, dtype=dt) * 255

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array    
    
    
class BatchNormLayer(Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, mode='low_mem',
    beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)
    Batch Normalization
    This layer implements batch normalization of its inputs, following [1]_:
    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta
    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.
    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.
    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.
    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.
    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.
    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.
    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.
    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer
    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem', beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized


def batch_norm(layer, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` and :class:`NonlinearityLayer` on top.
    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor.
    Returns
    -------
    BatchNormLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.
    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:
    >>> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))
    This introduces batch normalization right before its nonlinearity:
    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'BatchNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        from lasagne.layers import NonlinearityLayer
        layer = NonlinearityLayer(layer, nonlinearity)
    return layer

    
    
    