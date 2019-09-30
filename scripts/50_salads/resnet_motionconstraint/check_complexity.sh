#!/usr/bin/env bash
# Run master environment setup
source master_env.sh


# Check GPU if not using cloud server
if [ $IS_CLOUD = false ]
then
    if [ -z $1 ]
    then
        echo "Please specify device ID to use"
        exit
    fi
    GPUID=$1
fi


# Parameters to be changed
SPLIT="Split_1"
SNIPPET=16
#NET_NAME="resnet50_encoder"
NET_NAME="resnet50_motionconstraint"
LVL="mid"


# Generated parameters
LABELSDESC=$DATA_DIR"/data/50_salads_dataset/labels/actions_"$LVL"lvl.txt"


# Main command
cmd() {
    python3 src/check_complexity.py \
        --netname $NET_NAME \
        --labels_fname $LABELSDESC \
        --snippet_len $SNIPPET \
        --usemotionloss False
}

if [ $IS_CLOUD = true ]
then
    cmd
else
    CUDA_VISIBLE_DEVICES=$GPUID cmd
fi
