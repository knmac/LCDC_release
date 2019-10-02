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
SPLIT="Split_3"
NET_NAME="resnet50_motionconstraint"
SNIPPET=16
STRIDE=4
FRAMESKIP=1
MIX=1

EXP_NAME=$NET_NAME"_nomotion_g3"
LABELSDESC=$DATA_DIR"/data/GTEA/label_dict.txt"

N_EPOCHS=30         # number of epochs
DECAY_EPOCHS=10     # number of epoch before decaying
BATCH_SIZE=3        # number of samples per batch


# Generated parameters
DATA_NAME=$SPLIT"_snip"$SNIPPET"_stride"$STRIDE"_frs"$FRAMESKIP
if (( $MIX == 1 ))
then
    DATA_NAME=$DATA_NAME"_mix"
fi


# Compute number of iterations per epoch to save checkpoint every epoch
N_SAMPLES=$(head -n 1 $DATA_DIR"/data/GTEA/tfrecords/"$DATA_NAME"/train_n_samples.txt")
ITERS_PER_EPOCH=$(( N_SAMPLES / BATCH_SIZE )) 
NUM_ITER=$(( ITERS_PER_EPOCH * N_EPOCHS ))
DECAY_STEP=$(( ITERS_PER_EPOCH * DECAY_EPOCHS ))


# Main command
cmd() {
    python3 src/trainer.py \
        --recorddir $DATA_DIR"/data/GTEA/tfrecords/"$DATA_NAME \
        --logdir $RESULT_DIR"/logs/GTEA/"$DATA_NAME"/"$EXP_NAME \
        --record_regex "train_*" \
        --pretrained_model $PRETRAIN_DIR"/pretrain/Resnet50_deformNet_iter_145000_g3.pkl" \
        --pretrained_ext "pkl" \
        --labels_fname $LABELSDESC \
        --netname $NET_NAME \
        --datasetname "gtea" \
        --raw_height 405 \
        --raw_width 720 \
        --snippet_len $SNIPPET \
        --num_iter $NUM_ITER \
        --batch_size $BATCH_SIZE \
        --decay_steps $DECAY_STEP \
        --saving_freq $ITERS_PER_EPOCH \
        --optimizer "Momentum" \
        --wrapper "manual" \
        --usemotionloss False \
        --resume False
}

if [ $IS_CLOUD = true ]
then
    cmd
else
    CUDA_VISIBLE_DEVICES=$GPUID cmd
fi
