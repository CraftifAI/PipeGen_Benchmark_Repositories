#!/bin/bash

gdown --fuzzy https://drive.google.com/file/d/1TO-qGIt-vfbgzVMlbdrcSI1fOebWvX8S/view?usp=sharing
unzip -o ./bevfusion_models.zip
rm ./bevfusion_models.zip
cd data
gdown --fuzzy https://drive.google.com/file/d/1QIQ0YrVvfvqg0VKkCjyopeF1ox3SRUid/view?usp=sharing
unzip -o ./nuscenes_mini.zip
rm ./nuscenes_mini.zip