#!/usr/bin/env bash
if [ -z $1 ]
then
    echo "Please specify device ID to use"
    exit
fi
GPUID=$1

# Parameters to be changed
SPLIT="Split_5"
NET_NAME="resnet50_motionconstraint"
LVL="mid"
SNIPPET=16
STRIDE=4
FRAMESKIP=5
MIX=1
EXP_NAME=$NET_NAME"_nomotion_g3"


# Generated parameters
LABELSDESC="./data/50_salads_dataset/labels/actions_"$LVL"lvl.txt"
DATA_NAME=$SPLIT"_"$LVL"_snip"$SNIPPET"_stride"$STRIDE"_frs"$FRAMESKIP
if (( $MIX == 1 ))
then
    DATA_NAME=$DATA_NAME"_mix"
fi


# Main command
CUDA_VISIBLE_DEVICES=$GPUID python3 src/tester_fd.py \
    --datasetname "50salads" \
    --datadir "./data/50_salads_dataset/activity/"$LVL \
    --trainlogdir "./logs/50_salads/"$DATA_NAME"/"$EXP_NAME \
    --testlogdir "./logs/50_salads/"$DATA_NAME"/"$EXP_NAME"_test" \
    --split_fn "./data/50_salads_dataset/splits/"$SPLIT"/test.txt" \
    --labels_fname $LABELSDESC \
    --ext "png" \
    --netname $NET_NAME \
    --snippet_len $SNIPPET \
    --frameskip $FRAMESKIP \
    --batch_size 50 \
    --ckpt_fname "auto"
