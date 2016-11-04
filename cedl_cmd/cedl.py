#!/usr/bin/env python
'''
CellCognition Explorer - deep learning command-line extension
'''

import os
import sys

import cellh5
import h5py
import numpy
from numpy.lib.recfunctions import merge_arrays
import pandas

import logging
logger = logging.getLogger(__name__)

from autoencoders import Autoencoder, AdaGradTrainer, NestorovTrainer

import argparse

version = (1, 0, 0)

parser = argparse.ArgumentParser(prog="CellExplorer deep learning command line extension", version="{}.{}.{}".format(*version), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--verbose', action='store_true', help='verbose output', dest='loglevel')

parser.add_argument('--im_size', '-is', type=int, help='Size of the squared, cropped input images (in pixel)', default=60)    

subparsers = parser.add_subparsers()

train_parser   = subparsers.add_parser('train', help='Train a deep learning autoencoder from pre-processed image data stored in cellh5.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
train_parser.set_defaults(action="train")


train_parser.add_argument('cellh5_input', help='The cellh5 file as input for training an autoencoder.')
train_parser.add_argument('cellh5_mapping', help='Position mapping table (.txt file) with a "Group" Column to '
                                                 'indicate negative control conditions.' )
train_parser.add_argument('--autoencoder_architecture', '-ae', help='String describing the encoding layers of the autoencoder.\n'
                                                                    'Three layer types are supported:'
                                                                    '"cF.SA": convolutional layer with F filters of size SxS and activation A (A=r or A=s), '
                                                                    '"pP" max-pooling layer with pooling size of PxP\n, and '
                                                                    '"dN.D" fully-connected dense layer with output size N and additional drop-out layer with probability 0.D. '
                                                                    'Deep autoencoders can be constructed by concatenating layers with "_"'
                          
                          , default='c16.5r_p2_c32.3r_p2_d256.1r_d64.0s', dest='ae_arch')

train_parser.add_argument('--learner', '-l', choices=['nesterov', 'adagrad'], help='SGD variant. Choose between Nesterov momentum and AdaGrad updates.', default='addgrad')
train_parser.add_argument('--learning_rate', '-lr', type=float, help='Learning rate', default=0.05)
train_parser.add_argument('--momentum', '-m', type=float, help='Momentum for Nestorov updates', default=0.9)
train_parser.add_argument('--batchsize', '-bs' , type=int, help='Mini-batch size', default=128)
train_parser.add_argument('--corruption', '-c' , type=float, help='Initial corruption level of the denoising autoencoder', default=0.0)
train_parser.add_argument('--epochs', '-e' , type=int, help='Number of training epochs', default=16)
train_parser.add_argument('--nsamples', '-n' , type=int, help='The number of instances randomly sampled from negative control conditions', default=1000)
train_parser.add_argument('--neg_indicator', '-nd' , type=str, help='The token, which indicates a negative control condition in the mapping file', default='neg', dest='neg_condition')                                                 



predict_parser = subparsers.add_parser('encode', help='Encode image data (in cellh5) using a previously trained autoencoder. Result can be viewed and further analyzed using the CellExplorer GUI.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
predict_parser.set_defaults(action="encode")

predict_parser.add_argument('name', help='Name of the trained network (generated with option "train"' )
predict_parser.add_argument('cellh5_input', help='The cellh5 file as input for feature generation using an autoencoder.')
predict_parser.add_argument('cellh5_mapping', help='Position mapping table (.txt file), which contains all positions to be processed' )



class BaseReader(object):
    @staticmethod
    def normalize_galleries(gals):
        return gals.astype(numpy.float32) / 255.
    
    def write_galleriy_file(self, gals):
        output_file_name = self.get_output_file_name()
        
        norm_gals = self.normalize_galleries(gals[:self.pargs.nsamples, ...]) 
        
        with h5py.File(output_file_name, 'w') as h:
            h.create_dataset('galleries', data=norm_gals)
    
    def load_galleriy_file(self):
        output_file_name = self.get_output_file_name()
        with h5py.File(output_file_name, 'r') as h:
            res = h['galleries'].value[:, None, ...]
        return res
        
        
    def get_output_file_name(self):
        return '%s_galleries_%s_%dx%d_%dk.h5' % (self.name, self.pargs.neg_condition, self.pargs.im_size, 
                                                 self.pargs.im_size, self.pargs.nsamples / 1000)
    
    def iter_pos(self):
        raise NotImplementedError()

class Cellh5Reader(BaseReader):
    def __init__(self, name, pargs):
        self.pargs = pargs
        self.cellh5_input = pargs.cellh5_input
        self.cellh5_mapping = pargs.cellh5_mapping
        self.im_size = pargs.im_size
        
        self.name = name
        
        logger.debug("Check input files")
        if not os.path.isfile(self.cellh5_input): sys.exit("Error: Cellh5 file does not exist")
        if not os.path.isfile(self.cellh5_mapping): sys.exit("Error: Mapping file does not exist")
        
    def read_training_images(self):
        cr = self.cr = cellh5.CH5MappedFile(self.cellh5_input, 'r', cached=False)
        
        logger.debug("  Read plate mapping")
        cr.read_mapping(self.cellh5_mapping)
        
        neg_mapping = cr.mapping[cr.mapping["Group"] == self.pargs.neg_condition]
        
        result_galleries = []
        
        cnt = 0
        logger.debug("  Reading galleries from negative control conditions:")
        while True:
            rn = numpy.random.randint(len(neg_mapping))
            plate, well, site, gene = neg_mapping.iloc[rn][["Plate", "Well", "Site", "Gene Symbol"]]
            
            ch5_pos = cr.get_position(well, site)
            
            n_cells = len(ch5_pos.get_object_idx())
            logger.debug('     {} {} {} {} {}'.format(plate, well, site, gene, n_cells))
            if n_cells == 0:
                continue
            
            nr = numpy.random.randint(1, n_cells)
            gals = ch5_pos.get_gallery_image(range(nr), size=self.im_size).T.reshape((nr, self.im_size,self.im_size))
            result_galleries.append(gals)
            
            cnt += nr
            if cnt > self.pargs.nsamples:
                break
            
        return numpy.concatenate(result_galleries)
    
    def extract_galleries(self, pos, object_='primary__primary'):
        n_cells = len(pos.get_object_idx())
        cell_idx = range(n_cells)
        im_size = self.pargs.im_size
        
        img = None
        if len(cell_idx) > 0:
            img = pos.get_gallery_image(cell_idx, object_=('primary__primary'), size=im_size)
            img = img.reshape((im_size, n_cells,im_size,1)).transpose((1,3,0,2))
            
        return img
    
    def extract_geometry(self, pos, object_='primary__primary'):
        n_cells = len(pos.get_object_idx())
        if n_cells > 0:
            cell_idx = range(n_cells)
            
            centers = pos.get_center(cell_idx, object_=object_)
            bbox    = pos.get_feature_table(object_, 'bounding_box')[cell_idx]
            orient  = pos.get_feature_table(object_, 'orientation')[cell_idx]
            
            meta = numpy.zeros(n_cells, dtype=[('file','|S128' ), ('treatment','|S128'), ('label', int )])
            
            return merge_arrays((meta, centers, orient, bbox), asrecarray=True, flatten = True, usemask = False) 
        
    def iter_pos(self):
        self.cr = cr = cellh5.CH5MappedFile(self.cellh5_input, 'r', cached=False)
        cr.read_mapping(self.cellh5_mapping)
        for i, row in cr.mapping.iterrows():
            yield row, cr.get_position(row.Well, row.Site)
            
            
    def close(self):
        try:
            self.cr.close()
        except Exception as e:
            logger.warn("Warn: Error closing cellh5 input file\n" + str(e))
            
        

class Ch5Encoder(object):
    def __init__(self, ae, reader, writer):
        self.ae = ae
        self.reader = reader
        self.writer = writer
        
    def encode(self, data):
        return self.ae.encode(data)
        
    def write(self, content):
        logger.debug('  Write content')
    
    def run(self):
        for meta, pos in self.reader.iter_pos():
            logger.debug('     {} {} {}'.format(meta.Plate, meta.Well, meta.Site))
        
            ncell = pos.get_object_count()
            if ncell == 0:
                continue
            
            imgs = self.reader.extract_galleries(pos)
            geo  = self.reader.extract_geometry(pos)
            geo["file"] = self.reader.pargs.cellh5_input[:128]
            geo["treatment"] = meta["Gene Symbol"] if "Gene Symbol" in meta else "blub"
            geo["label"] = 0
            
            
            features = self.encode(self.reader.normalize_galleries(imgs))
            rec_features = features.view(dtype=numpy.dtype([("ch1-deep_learning_feature_{}".format(dl), 'float32') for dl in xrange(features.shape[1])]))
            
            self.writer.write_bbox(geo)
            self.writer.write_contours(geo, bb_size=self.reader.im_size)
            self.writer.write_features(rec_features)
            
            self.writer.write_galleries(imgs)
            
        image_width = pos['image']['channel'].shape[3]
        image_height = pos['image']['channel'].shape[4]
        
        self.writer.write_feature_groups()
        self.writer.gallery.dset.attrs["colors"] = ["#FFFFFF"]
        self.writer.gallery.dset.attrs["image_size"] = (image_width, image_height)
        
            
class StandardOutputWriter(object):
    BBOX_DTYPE = [('file', 'S128'), ('treatment', 'S128'), ('label', '<i4'), 
                  ('x', '<i4'), ('y', '<i4'), 
                  ('angle', '<f8'), ('eccentricity', '<f8'), 
                  ('left', '<i4'), ('right', '<i4'), ('top', '<i4'), ('bottom', '<i4')]
    
    FEATURE_DTYPE = [] # created dynamically
    
    CONTOUR_DTYPE = h5py.special_dtype(vlen=numpy.uint16)
    GALLERY_DTYPE = numpy.uint8

    def __init__(self, filename, im_size):
        self._fh = h5py.File(filename, 'w')
        self._fh.attrs["application"] = "CellExplorer_deep_learning_cmd-line-tool-{}.{}.{}".format(*version)
        self._fh.attrs["training_data"] = ["data",]
        self.data_grp = self._fh.create_group("data")
        
        self.bbox = self.create_writer("bbox", self.data_grp, self.BBOX_DTYPE)
        
        self.contour_grp = self.data_grp.create_group("contours")
        self.contour_grp.attrs["channels"] = ["Channel_1"]
        
        self.contours = self.create_writer("Channel_1", self.contour_grp, self.CONTOUR_DTYPE, kind="a", shape=('x', 2), grow_dim=0)
        self.gallery = self.create_writer("gallery", self.data_grp, self.GALLERY_DTYPE, kind="a", shape=(im_size, im_size,1,'x'), grow_dim=3)
        
        # not yet initializable
        self.features = None
        
    
    def create_writer(self, name, grp, dtype, kind='c', shape=None, grow_dim=0):
        if kind == "c":    
            return Hdf5IncrementalCompoundWriter(name, grp, dtype)
        elif kind == 'a':
            return Hdf5IncrementalArrayWriter(name, grp, dtype, shape, grow_dim)
        raise AttributeError("HDF5 Writer not supported. Choose Compound or Array, c or a.")
        
    def write_bbox(self, data):
        self.bbox.inc_write(data)
        
    def write_contours(self, bbox, bb_size):
        sh = bb_size/2
        x = numpy.c_[bbox['x']-sh, bbox["x"]+sh, bbox["x"]+sh, bbox["x"]-sh]
        y = numpy.c_[bbox['y']-sh, bbox["y"]-sh, bbox["y"]+sh, bbox["y"]+sh]
        
        self.contours.resize_for(x)
        # manual 
        self.contours.dset[self.contours.offset:self.contours.offset+len(x), 0] = x
        self.contours.dset[self.contours.offset:self.contours.offset+len(y), 1] = y
        self.contours.offset+=len(x)
        
    def write_galleries(self, gals):
        self.gallery.inc_write(gals.transpose())

    def write_features(self, data):
        if self.features is None:
            self.features = self.create_writer("features", self.data_grp, self.FEATURE_DTYPE)
        self.features.inc_write(data)
        
    def write_feature_groups(self):
        fg = self.data_grp.create_dataset("feature_groups", shape=(len(self.FEATURE_DTYPE),), dtype=[('feature', '|S64'), ('Simple1', '|S64')])
        fg['feature'] = zip(*self.FEATURE_DTYPE)[0]
        fg['Simple1'] = "Simple1"
          
    def close(self):
        self.bbox.finalize()
        self.contours.finalize()
        self.gallery.finalize()
        self.features.finalize()
        try:
            self._fh.close()
        except Exception as e:
            logger.warn("Error: Problem closing file handle: {}".format(str(e)))
        
class Hdf5IncrementalCompoundWriter(object):
    init_size = 1000    
    def __init__(self, object_name, obj_grp, dtype):
        self.obj_grp = obj_grp
        self.dtype = dtype
        self.offset = 0 
        self.object_name = object_name
       
        self.dset = self.obj_grp.create_dataset(self.object_name, shape=(self.init_size,), dtype=self.dtype, maxshape=(None,))
       
        
    def inc_write(self, data):
        if len(data) + self.offset > len(self.dset) :
            # resize
            self.dset.resize((len(data) + self.offset,))
            
        if len(data.shape) == 2:
            data = data[:,0]
        
        self.dset[self.offset:self.offset+len(data)] = data
        self.offset+=len(data)
        
    def finalize(self):
        self.dset.resize(self.offset, axis=0)
        
class Hdf5IncrementalArrayWriter(object):
    init_size = 1000    
    def __init__(self, object_name, obj_grp, dtype, shape, grow_dim=0):
        self.obj_grp = obj_grp
        self.dtype = dtype
        self.offset = 0 
        self.object_name = object_name
        self.grow_dim = grow_dim
        
        init_shape = list(shape)
        init_shape[grow_dim] = self.init_size
        
        maxshape = list(shape)
        maxshape[grow_dim] = None
        self.dset = self.obj_grp.create_dataset(self.object_name, shape=init_shape, dtype=self.dtype, maxshape=maxshape)
        
    def resize_for(self, data):
        if data.shape[self.grow_dim] + self.offset > self.dset.shape[self.grow_dim] :
            # resize
            new_shape = list(self.dset.shape)
            new_shape[self.grow_dim] = self.dset.shape[self.grow_dim] + data.shape[self.grow_dim]
            self.dset.resize(new_shape)
        
    def finalize(self):
        final_shape = list(self.dset.shape)
        final_shape[self.grow_dim] = self.offset
        self.dset.resize(final_shape)
        
    def inc_write(self, data):
        self.resize_for(data)
        
        index = [slice(None, None, None), ] * data.ndim
        index[self.grow_dim] = slice(self.offset, self.offset + data.shape[self.grow_dim], None)
        self.dset[tuple(index)] = data
        
        self.offset+=data.shape[self.grow_dim]
    
        
def train(args):
    name = "{}_{}".format(os.path.splitext(os.path.basename(args.cellh5_input))[0], args.ae_arch) 
    logger.info("Training: '{}'".format(name))
    
    logger.info("Init reader")
    cr = Cellh5Reader(name, args)
    
    logger.info("Open cellh5 file")
    galleries = cr.read_training_images()
    
    logger.info("Write images to output file")
    cr.write_galleriy_file(galleries)
    
    logger.info("Load images from output file")
    galleries = cr.load_galleriy_file()
    
    logger.info("Init deep learning network")
    ae = Autoencoder(cr.name, (1, cr.pargs.im_size, cr.pargs.im_size), denoising=cr.pargs.corruption)
    
    logger.info("Set deep learning network architecture to: '{}'".format(cr.pargs.ae_arch))
    arch = cr.pargs.ae_arch.replace("_", " ")
    ae.init_from_string(arch)
    
    logger.info("Configure trainer: '{}'".format(cr.pargs.learner))
    if cr.pargs.learner == "nesterov":
        trainer = NestorovTrainer(epochs=cr.pargs.epochs, 
                                  batchsize=cr.pargs.batchsize,  
                                  learning_rate=cr.pargs.learning_rate, 
                                  momentum=cr.pargs.momentum
                                  )
    else:
        trainer = AdaGradTrainer(epochs=cr.pargs.epochs, 
                                 batchsize=cr.pargs.batchsize,  
                                 learning_rate=cr.pargs.learning_rate, 
                                 )
        
    logger.info("Training: (can take hours...)")
    trainer(ae).fit(galleries, None)
    
    logger.info("Save deep learning network as '{}' (use this for encode)".format(name))
    ae.save()

def predict(args):
    logger.info("Init reader")
    cr = Cellh5Reader(args.name, args)
    
    
    logger.info("Read autoencoder model")
    try:
        ae = Autoencoder.load(args.name)
    except Exception as e:
        logging.error("Error loading autoencoder {}".format(str(e)))
        sys.exit("Error: Cellh5 file does not exist")
    
    output_file = "{}.hdf".format(args.name)
    logger.info("Init output writer -> {}".format(output_file))
    wr = StandardOutputWriter(output_file, args.im_size)
    wr.FEATURE_DTYPE = [("ch1-deep_learning_feature_{}".format(dl), 'float32') for dl in xrange(ae.get_code_size())]
    
    logger.info("Encode: (can take hours...)")
    encoder = Ch5Encoder(ae, cr, wr)
    encoder.run()
    
    wr.close()
    cr.close()
    
    logger.info("Output file created. Open CellCognition Explorer GUI and open '{}'".format(output_file))
    
def main(args):
    logging.basicConfig(level=args.loglevel or logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.action == "train":         
        train(args)
    elif args.action == "encode":
        predict(args)    
    logger.debug(" --- The End ---")
    

if __name__ == '__main__':
    main(parser.parse_args())

    