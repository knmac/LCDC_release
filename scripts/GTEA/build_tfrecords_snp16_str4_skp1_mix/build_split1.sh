#!/usr/bin/env bash
# Build tfrecords for GTEA dataset


# Parameters to be changed
ROOT="./data/GTEA"
DATASET_NAME="gtea"
SPLIT="Split_1"
SNIPPET=16
STRIDE=4
FRAMESKIP=1
MIX=1
PURITY=0.6


# Generated parameters
LABELSDESC=$ROOT"/label_dict.txt"
OUTPUT_DIR=$ROOT"/tfrecords/"$SPLIT"_snip"$SNIPPET"_stride"$STRIDE"_frs"$FRAMESKIP
if (( $MIX == 1 ))
then
    OUTPUT_DIR=$OUTPUT_DIR"_mix"
fi


# Main command
python ./src/build_tfrecords.py \
    --datasetname $DATASET_NAME \
    --datadir $ROOT"/activity" \
    --outputdir $OUTPUT_DIR \
    --labelsdesc $LABELSDESC \
    --splitlist $ROOT"/splits/"$SPLIT"/train.txt" \
    --mode "train" \
    --output_pattern "train" \
    --ext "png" \
    --snippet_len $SNIPPET \
    --frameskip $FRAMESKIP \
    --stride $STRIDE \
    --mix $MIX \
    --purity $PURITY \
    --dummy 0
