#!/bin/bash

output_dir="model_output_"`date +"%Y%m%d%H%M%S"`
# output_dir="model_output" # test
phase="train"
training_set="./datasets/tfrecords/VOC0712.tfrecords"
batch_size=1 # MemNet_M6R6, batch_size is 1.
training_steps=100000
summary_steps=50
checkpoint_steps=1000
save_steps=500


python main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --training_set="$training_set" \
               --batch_size="$batch_size" \
               --training_steps="$training_steps" \
               --summary_steps="$summary_steps" \
               --checkpoint_steps="$checkpoint_steps" \
               --save_steps="$save_steps"
