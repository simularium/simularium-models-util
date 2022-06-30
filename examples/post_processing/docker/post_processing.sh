#!/bin/bash

# copy parameter file
case ${SIMULATION_TYPE} in
	AWS)
		INPUT_FILE_PATH="${S3_INPUT_URL}parameters/"
		OUTPUT_FILE_PATH="${S3_INPUT_URL}outputs/"
        PARAMS_COL_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
		INPUT_FILE_NAME="${PARAM_SET_NAME}.xlsx"
		aws s3 cp $INPUT_FILE_PATH$INPUT_FILE_NAME input.xlsx
	;;
	LOCAL)
		INPUT_FILE_PATH="/working/"
		OUTPUT_FILE_PATH="/working/"
        PARAMS_COL_INDEX=$JOB_ARRAY_INDEX
		INPUT_FILE_NAME="${PARAM_SET_NAME}.xlsx"
		cp $INPUT_FILE_PATH$INPUT_FILE_NAME input.xlsx
	;;
esac

LOCAL_LOGS_PATH="logs/${PARAM_SET_NAME}_${PARAMS_COL_INDEX}_logs.txt"
mkdir ./logs
touch $LOCAL_LOGS_PATH
PARAMS_COL_INDEX=$((${PARAMS_COL_INDEX} + 1))

# Run the model
python post_processing.py input.xlsx $PARAMS_COL_INDEX $PARAM_SET_NAME > $LOCAL_LOGS_PATH
EXIT_CODE=$?

# Save output files
case ${SIMULATION_TYPE} in
	AWS)
		aws s3 sync ./outputs $OUTPUT_FILE_PATH
	;;
	LOCAL)
		cd outputs
		cp *.h5 $OUTPUT_FILE_PATH
		cp *.simularium $OUTPUT_FILE_PATH
		cp *.dat $OUTPUT_FILE_PATH
	;;
esac
