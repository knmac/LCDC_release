#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Preprocessing data for GTEA dataset
# -----------------------------------------------------------------------------

DATA_ROOT="./data/GTEA"                     # root directory
FRAME_DIR=${DATA_ROOT}"/png"                # extracted frames dir
LABEL_DIR=${DATA_ROOT}"/labels"             # label dir
OUTPUT_DIR=${DATA_ROOT}"/activity"          # segmented frames output dir
LABEL_DICT=${DATA_ROOT}"/label_dict.txt"    # label dictionary
IMG_EXT=".png"                              # extracted frame extension
BG_LBL="background"                         # background label


# Seqgment activities using the data's annotation------------------------------
echo "Segmenting activities..."
python ./src/data_utils/segment_activities_gtea.py \
    --framedir ${FRAME_DIR} \
    --labeldir ${LABEL_DIR} \
    --outputdir ${OUTPUT_DIR} \
    --labeldict_pth ${LABEL_DICT} \
    --ext ${IMG_EXT} \
    --bg_lbl ${BG_LBL}
