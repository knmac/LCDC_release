#!/usr/bin/env bash
# Train rDeRF-Net with resnet50 backbone and deformable layers 5a, 5b, 5c.
# Initialize from pretrained weights of TF_Deformable_Net

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
SPLIT="Split_4"
NET_NAME="resnet50_motionconstraint"
LVL="mid"
SNIPPET=16
STRIDE=4
FRAMESKIP=5
MIX=1

EXP_NAME=$NET_NAME"_nomotion_g1"

N_EPOCHS=30         # number of epochs
DECAY_EPOCHS=10     # number of epoch before decaying
BATCH_SIZE=3        # number of samples per batch


# Generated parameters
LABELSDESC=$DATA_DIR"/data/50_salads_dataset/labels/actions_"$LVL"lvl.txt"
DATA_NAME=$SPLIT"_"$LVL"_snip"$SNIPPET"_stride"$STRIDE"_frs"$FRAMESKIP
if (( $MIX == 1 ))
then
    DATA_NAME=$DATA_NAME"_mix"
fi


# Compute number of iterations per epoch to save checkpoint every epoch
N_SAMPLES=$(head -n 1 $DATA_DIR"/data/50_salads_dataset/tfrecords/"$DATA_NAME"/train_"$LVL"_n_samples.txt")
ITERS_PER_EPOCH=$(( N_SAMPLES / BATCH_SIZE )) 
NUM_ITER=$(( ITERS_PER_EPOCH * N_EPOCHS ))
DECAY_STEP=$(( ITERS_PER_EPOCH * DECAY_EPOCHS ))


# Main command
cmd() {
    python3 src/trainer.py \
        --recorddir $DATA_DIR"/data/50_salads_dataset/tfrecords/"$DATA_NAME \
        --logdir $RESULT_DIR"/logs/50_salads/"$DATA_NAME"/"$EXP_NAME \
        --record_regex "train_"$LVL"_*" \
        --pretrained_model $PRETRAIN_DIR"/pretrain/Resnet50_deformNet_iter_145000_g1.pkl" \
        --pretrained_ext "pkl" \
        --labels_fname $LABELSDESC \
        --netname $NET_NAME \
        --datasetname "50salads" \
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
