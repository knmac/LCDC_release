#!/usr/bin/env bash
if [ -z $1 ]
then
    echo "Please specify device ID to use"
    exit
fi
GPUID=$1

# Parameters to be changed
SPLIT="Split_1"
NET_NAME="resnet50_motionconstraint"
SNIPPET=16
STRIDE=4
FRAMESKIP=1
MIX=1
EXP_NAME=$NET_NAME"_nomotion_g0"
LABELSDESC="./data/GTEA/label_dict.txt"


# Generated parameters
DATA_NAME=$SPLIT"_snip"$SNIPPET"_stride"$STRIDE"_frs"$FRAMESKIP
if (( $MIX == 1 ))
then
    DATA_NAME=$DATA_NAME"_mix"
fi


# Main command
CUDA_VISIBLE_DEVICES=$GPUID python3 src/tester_fd.py \
    --datasetname "gtea" \
    --datadir "./data/GTEA/activity/" \
    --trainlogdir "./logs/GTEA/"$DATA_NAME"/"$EXP_NAME \
    --testlogdir "./logs/GTEA/"$DATA_NAME"/"$EXP_NAME"_test" \
    --split_fn "./data/GTEA/splits/"$SPLIT"/test.txt" \
    --labels_fname $LABELSDESC \
    --ext "png" \
    --netname $NET_NAME \
    --snippet_len $SNIPPET \
    --frameskip $FRAMESKIP \
    --batch_size 50 \
    --ckpt_fname "auto"
