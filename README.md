# CellCognition Explorer - deep learning command-line extension


## Prerequisites

To run cedl.py, please unzip the demo data 

* cellH5-file    : CecogExCedlDemo_plate_1.ch5
* full-mapping   : CecogExCedlDemo_plate_1_mapping_full.txt
* encode-mapping : CecogExCedlDemo_plate_1_mapping_encode.txt
                       (contains only a subset of positions)
into the "data" folder of the cedl command-line tool.
 
## Help pages

$ python cedl.py --help

$ python cedl.py train --help

$ python cedl.py encode --help
 
 
## Training

$ python cedl.py --im_size 60 --verbose train --learner nesterov --autoencoder_architecture c16.5r_p2_c32.3r_p2_d256.1r_d64.0s ../data/CecogExCedlDemo_plate_1.ch5 ../data/CecogExCedlDemo_plate_1_mapping_full.txt

This will create an deep learning autoencoder model called "CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s".
 This name is needed in the enocding step:

## Encoding

$ python cedl.py --im_size 60 --verbose encode CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s ../data/CecogExCedlDemo_plate_1.ch5 ../data/CecogExCedlDemo_plate_1_mapping_encode.txt

Will create "CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s.hdf"

## Visualization and Novelty detection

Open CellCognition Explorer GUI load "CecogExCedlDemo_plate_1_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s_c16.5r_p2_c32.3r_p2_d256.1r_d64.0s.hdf"
 

