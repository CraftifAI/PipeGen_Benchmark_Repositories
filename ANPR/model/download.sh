#!/bin/bash

mkdir -p detect && cd detect
gdown --fuzzy https://drive.google.com/file/d/1ozOt5StNo4G9fQ73Uf_fEiwUmd1IVVsS/view?usp=sharing

cd ..

mkdir -p ocr && cd ocr
gdown --fuzzy https://drive.google.com/file/d/1rkyP-l3YQF4VtM2GCIIckkxKe4pA_5_i/view?usp=sharing
# The OCR model is useless without its alphabet/region config; application.py falls back to
# fetching this over the network, so grab it here too and keep the run fully offline-capable.
curl -fsSL -o plate_ocr_config.yaml \
  https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml

cd ..
