#!/usr/bin/env bash
if [ -z $1 ]
then
    echo "Please specify device ID to use"
    exit
fi


SPLIT=2
LVL="mid"
NETNAME="resnet50_motionconstraint"
PRETRAIN_DIR="./logs/50_salads/Split_"$SPLIT"_mid_snip16_stride4_frs5_mix/resnet50_motionconstraint_nomotion_g0_test/best"
OUTPUTDIR="./data/50_salads_dataset/tcnfeat"$LVL"/"$NETNAME"_nomotion_g0/Split_"$SPLIT

CKPT_NAME=$( ls $PRETRAIN_DIR | grep .index )
CKPT_NAME=${CKPT_NAME//.index/}


CUDA_VISIBLE_DEVICES=$1 python3 ./src/extract4tcn.py \
    --netname $NETNAME \
    --datasetname "50salads" \
    --segmented_dir "./data/50_salads_dataset/activity/mid" \
    --lbl_dict_pth "./data/50_salads_dataset/labels/actions_midlvl.txt" \
    --outputdir $OUTPUTDIR \
    --pretrained_model $PRETRAIN_DIR"/"$CKPT_NAME \
    --featname "fc_new_2" \
    --snippet_len 16 \
    --batch_size 50 \
    --frameskip 5 \
    --stride 2
