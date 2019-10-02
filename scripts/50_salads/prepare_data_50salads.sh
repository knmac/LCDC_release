#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Preprocessing data for 50 salads dataset
# -----------------------------------------------------------------------------
DATA_ROOT="./data/50_salads_dataset"            # root directory
VID_DIR=${DATA_ROOT}"/rgb"                      # input video dir
FRAME_DIR=${DATA_ROOT}"/frames"                 # extracted frames dir
LEVEL="mid"                                     # level of granularity
ANNO_DIR=${DATA_ROOT}"/dense_anno/"${LEVEL}     # annotation dir
OUTPUT_DIR=${DATA_ROOT}"/activity/"${LEVEL}     # segmented frames output dir
VID_EXT=".avi"                                  # video extension
IMG_EXT=".png"                                  # extracted frame extension


# Extract all frames from videos using ffmpeg----------------------------------
if [ ! -d ${FRAME_DIR} ]; then
    echo "Extracting frames..."

    # make output dir
    mkdir -p ${FRAME_DIR}

    # process each video in input dir
    for file in ${VID_DIR}/*${VID_EXT}; do
        # retrieve only the video without extension or parent dir
        vid=${file//${VID_EXT}/""}
        vid=${vid//$VID_DIR"/"/""}
        echo $vid

        # make sub dir for the video being processed
        out_dir=${FRAME_DIR}"/"${vid}
        mkdir -p ${out_dir}

        # extract frame using ffmpeg
        ffmpeg -i ${file} ${out_dir}"/frame_%07d"${IMG_EXT} -hide_banner
    done
fi

# Seqgment activities using the data's annotation------------------------------
echo "Segmenting activities..."
python ./src/data_utils/segment_activities_50salads.py \
    --framedir ${FRAME_DIR} \
    --annodir ${ANNO_DIR} \
    --outputdir ${OUTPUT_DIR} \
    --ext ${IMG_EXT} \
    --level ${LEVEL}

# Clean up---------------------------------------------------------------------
#rm -r ${FRAME_DIR}
