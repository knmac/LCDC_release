#!/usr/bin/env bash
# Build tfrecords for 50 salad dataset


# Parameters to be changed
ROOT="./data/50_salads_dataset"
DATASET_NAME="50salads"
SPLIT="Split_1"
LVL="mid"
SNIPPET=16
STRIDE=4
FRAMESKIP=5
MIX=1
PURITY=0.7


# Generated parameters
LABELSDESC=${ROOT}"/labels/actions_"${LVL}"lvl.txt"
OUTPUT_DIR=${ROOT}"/tfrecords/"${SPLIT}"_"${LVL}"_snip"${SNIPPET}"_stride"${STRIDE}"_frs"${FRAMESKIP}
if (( ${MIX} == 1 )); then
    OUTPUT_DIR=${OUTPUT_DIR}"_mix"
fi


# Main command
python ./src/build_tfrecords.py \
    --datasetname ${DATASET_NAME} \
    --datadir ${ROOT}"/activity/"${LVL} \
    --outputdir ${OUTPUT_DIR} \
    --labelsdesc ${LABELSDESC} \
    --splitlist ${ROOT}"/splits/"${SPLIT}"/train.txt" \
    --mode "train" \
    --output_pattern "train_"${LVL} \
    --ext "png" \
    --snippet_len ${SNIPPET} \
    --frameskip ${FRAMESKIP} \
    --stride ${STRIDE} \
    --mix ${MIX} \
    --purity ${PURITY} \
    --dummy 0
