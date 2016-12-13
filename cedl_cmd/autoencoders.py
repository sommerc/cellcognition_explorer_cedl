"""


"""

import os
import sys; sys.setrecursionlimit(50000)
import pickle

import numpy; np = numpy
from PIL import Image

import lasagne
from lasagne.layers.noise import DropoutLayer
from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer, Upscale2DLayer
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, PrintLog
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast 
except:
    Conv2DLayerFast = Conv2DLayer
    MaxPool2DLayerFast = MaxPool2DLayer
    
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum, adagrad

from utils import TransposedDenseLayer, act, pprint_layers, tile_raster_images


class Autoencoder(object):
    """
    """
    
    def __init__(self, name, input_shape, denoising=0.0, verbose=2):
        self.name = name
        self.input_shape = input_shape
        self.n_channel = input_shape[0]
        self.pixel_size = input_shape[1]
        
        self.denoising = denoising
        self.verbose=verbose
        
        self._output_dir = None
        
        self.ae = None
        self.trainers = []
        
        
    def init_from_layer(self, last_layer):
        self.set_output_dir(self.name)
        self._nn = NeuralNet(last_layer, verbose=self.verbose, regression=True)
        pprint_layers(last_layer)
    
    def check_arch_string(self, arch_string):
        return
    
    def init_from_string(self, arch_string):
        self.check_arch_string(arch_string)
        self.set_output_dir(self.name)

        self.input_layer = InputLayer(shape=(None,)+ self.input_shape, name='input')
        
        net = self.input_layer
        net = DropoutLayer(net, p=self.denoising)
        
        # Encoder
        self._layers = []
        self.conv_layers = []
        self.encode_layers = []
        self.middle_pool_size = self.pixel_size
        for layer_string in arch_string.split():
            if layer_string.startswith("c"):
                non_lin = layer_string[-1]
                num_filter, filter_size = map(int,layer_string[1:-1].split("."))
                net = Conv2DLayerFast(net, num_filters=num_filter, filter_size=filter_size, nonlinearity=act(non_lin), pad='same')
                self.conv_layers.append(net)
                
            elif layer_string.startswith("p"):
                pool_size = int(layer_string[1])
                net = MaxPool2DLayerFast(net, pool_size=(pool_size, pool_size))
                self.middle_pool_size /= 2
                
            elif layer_string.startswith("d"):
                non_lin = layer_string[-1]
                num_units, noise = map(int,layer_string[1:-1].split("."))
#                 net = DenseLayer(batch_norm(lasagne.layers.dropout(net, p=noise*0.1)), num_units=num_units, nonlinearity=act(non_lin))
                net = DenseLayer(lasagne.layers.dropout(net, p=noise*0.1), num_units=num_units, nonlinearity=act(non_lin))
                self.encode_layers.append(net)
                
            self._layers.append(net)
             
        # Decoder
        for lyr in self._layers[::-1][:-1]:
            if isinstance(lyr, (Conv2DLayerFast,)):
                net = Conv2DLayerFast(net, num_filters=lyr.input_layer.output_shape[1], filter_size=(lyr.filter_size,lyr.filter_size), nonlinearity=lyr.nonlinearity, pad='same' )
                
            elif isinstance(lyr, (MaxPool2DLayerFast,)):
                if len(net.output_shape) == 2:
                    net = ReshapeLayer(net, shape=([0], lyr.input_layer.num_filters, self.middle_pool_size, self.middle_pool_size))
                    
                net = Upscale2DLayer(net, scale_factor=lyr.pool_size)
                
            elif isinstance(lyr, (DenseLayer,)):
                net = TransposedDenseLayer(net, num_units=numpy.prod(lyr.input_layer.input_shape[1:]), W=lyr.W, nonlinearity=lyr.nonlinearity)
            
                
        lyr = self._layers[0]
        if isinstance(lyr, (Conv2DLayerFast,)):
            net = Conv2DLayerSlow(net, num_filters=self.n_channel, filter_size=(lyr.filter_size, lyr.filter_size), nonlinearity=lyr.nonlinearity, pad='same' )
            
        elif isinstance(lyr, (MaxPool2DLayerFast,)):
            channels = 1
            if len(net.output_shape) == 2 and isinstance(lyr.input_layer, (Conv2DLayerFast, Conv2DLayerSlow)):
                channels = lyr.input_layer.num_filters
            net = ReshapeLayer(net, shape=([0], channels, self.middle_pool_size, self.middle_pool_size))
            
                
            net = Upscale2DLayer(net, scale_factor=lyr.pool_size)
            
        elif isinstance(lyr, (DenseLayer,)):
            net = TransposedDenseLayer(net, num_units=numpy.prod(lyr.input_layer.input_shape[1:]), W=lyr.W, nonlinearity=lyr.nonlinearity)
            
        net = ReshapeLayer(net, name='output', shape=([0], -1))
        
        pprint_layers(net)
        self._nn = NeuralNet(net, verbose=self.verbose, regression=True, objective_loss_function=squared_error)

    def set_param(self, key, value):
        setattr(self._nn, key, value)
        
    def get_param(self, key):
        return getattr(self._nn, key)
    
    def add_param(self, key, value):
        param = self.get_param(key)
        param.append(value)
        
    def set_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._output_dir = output_dir
        
    def output(self, filename=None, path=None):
        if not path is None:
            if not os.path.exists(os.path.join(self._output_dir, path)):
                os.makedirs(os.path.join(self._output_dir, path))
        else:
            path = "."
        if filename is None:
            return os.path.join(self._output_dir, path)
        else:
            return os.path.join(self._output_dir, path, filename)
    
    def save(self):
        self._nn.on_epoch_finished = []
        with open(self.output(filename='_model.pickle'), 'wb') as f:
            pickle.dump(self, f, -1)
    
    @staticmethod
    def load(name):
        m_file = os.path.join(name, '_model.pickle')
        if not os.path.isfile(m_file): raise(IOError("Autoencoder model file does not exist: {}".format(m_file)))
        with open(m_file, 'rb') as f:
            inst = pickle.load(f)       
        inst._nn.on_epoch_finished=[PrintLog()] 
        return inst
    
    def fit(self, X, y):
        self._nn.fit(X,X.reshape(X.shape[0], -1))
        
    def encode(self, input):
        return get_output(self.encode_layers[-1], input, deterministic=True).eval()
        
    def get_code_size(self):
        return self.encode_layers[-1].output_shape[1]
    
class BaseTrainer(object):
    DataBatchIterator = BatchIterator
    def __init__(self, epochs, batchsize):
        self.epochs = epochs
        self.batchsize = batchsize
        
    def __call__(self, ae):
        ae.trainers.append(self)
        ae.set_param('max_epochs', self.epochs)
        ae.set_param('batch_iterator_train', self.DataBatchIterator(self.batchsize))
        
        return ae
    
class NestorovTrainer(BaseTrainer):
    def __init__(self, learning_rate=0.01, momentum=0.9, **kwargs):
        super(NestorovTrainer, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        
    def __call__(self, ae):
        super(NestorovTrainer, self).__call__(ae)
        
        ae.set_param('update', nesterov_momentum)
        ae.set_param('update_learning_rate', self.learning_rate)
        ae.set_param('update_momentum', self.momentum)
        
        return ae
    
class AdaGradTrainer(BaseTrainer):
    def __init__(self, learning_rate=0.01, **kwargs):
        super(AdaGradTrainer, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        
    def __call__(self, ae):
        super(AdaGradTrainer, self).__call__(ae)
        
        ae.set_param('update', adagrad)
        ae.set_param('update_learning_rate', self.learning_rate)
        
        return ae
    
    
def visualize_reconstruction(outputdir, data):
    tile_size = int( numpy.sqrt(data.shape[0]))
    Image.fromarray(tile_raster_images(X=data.reshape(data.shape[0], -1),
                                   img_shape=(data.shape[2], data.shape[3]), tile_shape=(tile_size, tile_size),
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True
                                   )).save(os.path.join(outputdir, "reco__.png"))
    def tmp(nn_, train_history):
        epoch = train_history[-1]['epoch']
        
        reco = nn_.predict(data)
        
        Image.fromarray(tile_raster_images(X=reco.reshape(data.shape[0], -1),
                                   img_shape=(data.shape[2], data.shape[3]), tile_shape=(tile_size, tile_size),
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True
                                   )).save(os.path.join(outputdir, "reco_%04d.png" % epoch))
                                   
    return tmp
        
        
def visualize_dense_weights(outputdir, lyr):
    img_shape = lyr.input_shape[2]
    tile_size = int(numpy.sqrt(lyr.num_units))
    def tmp(nn_, train_history):
        epoch = train_history[-1]['epoch']
        
        filter_ = lyr.W.get_value(True).T
        filter_ = filter_.reshape((filter_.shape[0], img_shape, img_shape, -1))
        
        for f in xrange(filter_.shape[-1]):
            Image.fromarray(tile_raster_images(X=filter_[..., f].reshape((filter_.shape[0], -1)),
                                       img_shape=(img_shape, img_shape), tile_shape=(tile_size, tile_size),
                                       tile_spacing=(1, 1), scale_rows_to_unit_interval=True
                                       )).save(os.path.join(outputdir, "weights_n%04d_f%d.png" % (epoch, f)))
        
    return tmp

def visualize_convolution_response(outputdir, lyr, data):
    tile_size = int( numpy.sqrt(data.shape[0]))
    Image.fromarray(tile_raster_images(X=data.reshape(data.shape[0], -1),
                                   img_shape=(data.shape[2], data.shape[3]), tile_shape=(tile_size, tile_size),
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True
                                   )).save(os.path.join(outputdir, "resp__.png"))
    def tmp(nn_, train_history):
        epoch = train_history[-1]['epoch']
         
         
        resp = get_output(lyr, data, deterministic=True).eval()
        tile_size = int( numpy.sqrt(resp.shape[1]))
        for d in xrange(resp.shape[0]):
            Image.fromarray(tile_raster_images(X=resp[d,...].reshape((resp.shape[1], -1)),
                                       img_shape=(resp.shape[2], resp.shape[3]), tile_shape=(tile_size, tile_size),
                                       tile_spacing=(1, 1), scale_rows_to_unit_interval=True
                                       )).save(os.path.join(outputdir, "resp_%04d_d%d.png" % (epoch,d)))
         
    return tmp
    
    
        

    
    
    
    

    
    
    
    
