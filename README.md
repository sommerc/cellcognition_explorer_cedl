# CellCognition Explorer - Deep Learning Extension
CellCognition Explorer Deep Learning learning Extension enables the unsupervised learning of cellular features directly based on image data. Only the rough bounding-box is required.

The Extension can be operated by a graphical user interface or by command line.

##


## Prerequisites

To run cedl.py, please unzip the demo data 

* cellH5-file    : CecogExCedlDemo_plate_1.ch5
* full-mapping   : CecogExCedlDemo_plate_1_mapping_full.txt
* encode-mapping : CecogExCedlDemo_plate_1_mapping_encode.txt
                       (contains only a subset of positions)
into the "data" folder of the cedl command-line tool.

## The graphical user interface
### Using installation based on docker
Downlaod ...

## The command line
### Help pages
```
$ python cedl.py --help

$ python cedl.py train --help

$ python cedl.py encode --help
``` 
 
### Training

```
$ python cedl.py --im_size 60 --verbose train --learner nesterov --autoencoder_architecture c16.5r_p2_c32.3r_p2_d256.1r_d64.0s ../data/CecogExCedlDemo_plate_1.ch5 ../data/CecogExCedlDemo_plate_1_mapping_full.txt
```

This will create an deep learning autoencoder model called "CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s".
 This name is needed in the enocding step:

### Encoding
```
$ python cedl.py --im_size 60 --verbose encode CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s ../data/CecogExCedlDemo_plate_1.ch5 ../data/CecogExCedlDemo_plate_1_mapping_encode.txt
```
Will create `"CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s.hdf"`

## Visualization and Novelty detection

Open CellCognition Explorer GUI and load `"CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s.hdf"`
 
