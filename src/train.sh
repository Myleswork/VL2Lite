#!/bin/bash
python3 train.py -m data/attributes=2_NABirds,3_DTD,4_OxfordIIITPet,5_StanfordDogs,6_StanfordCars,7_CALTECH101,8_CALTECH256,9_GTSRB model.use_teacher=true,false
python3 src/train.py -m data/attributes=6_StanfordCars model.use_teacher=true data.batch_size=256 data.num_workers=16 trainer.devices=1
#0_CUB_200_2011,1_FGVC_AIRCRAFT,
