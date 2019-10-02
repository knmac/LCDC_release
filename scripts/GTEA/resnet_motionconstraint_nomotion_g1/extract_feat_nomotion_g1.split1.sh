#!/usr/bin/env bash
if [ -z $1 ]
then
    echo "Please specify device ID to use"
    exit
fi


SPLIT=1
NETNAME="resnet50_motionconstraint"
PRETRAIN_DIR="./logs/GTEA/Split_"$SPLIT"_snip16_stride4_frs1_mix/"$NETNAME"_nomotion_g1_test/best"
OUTPUTDIR="./data/GTEA/tcnfeat/"$NETNAME"_nomotion_g1/Split_"$SPLIT

CKPT_NAME=$( ls $PRETRAIN_DIR | grep .index )
CKPT_NAME=${CKPT_NAME//.index/}


CUDA_VISIBLE_DEVICES=$1 python3 ./src/extract4tcn.py \
    --netname $NETNAME \
    --datasetname "gtea" \
    --segmented_dir "./data/GTEA/activity" \
    --lbl_dict_pth "./data/GTEA/label_dict.txt" \
    --outputdir $OUTPUTDIR \
    --pretrained_model $PRETRAIN_DIR"/"$CKPT_NAME \
    --featname "fc_new_2" \
    --snippet_len 16 \
    --batch_size 50 \
    --frameskip 1 \
    --stride 5
