#!/usr/bin/env bash
if [ -z $1 ]
then
    echo "Please specify device ID to use"
    exit
fi


SPLIT=3
LVL="eval"
NETNAME="resnet50_motionconstraint"
PRETRAIN_DIR="./logs/50_salads/Split_"$SPLIT"_"$LVL"_snip16_stride4_frs5_mix/"$NETNAME"_nomotion_g0_test/best"
OUTPUTDIR="./data/50_salads_dataset/tcnfeat"$LVL"/"$NETNAME"_nomotion_g0/Split_"$SPLIT

CKPT_NAME=$( ls $PRETRAIN_DIR | grep .index )
CKPT_NAME=${CKPT_NAME//.index/}


CUDA_VISIBLE_DEVICES=$1 python3 ./src/extract4tcn.py \
    --netname $NETNAME \
    --datasetname "50salads" \
    --segmented_dir "./data/50_salads_dataset/activity/"$LVL \
    --lbl_dict_pth "./data/50_salads_dataset/labels/actions_"$LVL"lvl.txt" \
    --outputdir $OUTPUTDIR \
    --pretrained_model $PRETRAIN_DIR"/"$CKPT_NAME \
    --featname "fc_new_2" \
    --snippet_len 16 \
    --batch_size 50 \
    --frameskip 5 \
    --stride 2
