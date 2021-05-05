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
python actin.py input.xlsx $PARAMS_COL_INDEX $PARAM_SET_NAME > $LOCAL_LOGS_PATH
EXIT_CODE=$?

# Save output files
if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 88888888 ]
then
	# if totally succeeded, or if simulation succeeded and visualization failed
	# results, checkpoints, and logs
	case ${SIMULATION_TYPE} in
		AWS)
            aws s3 sync ./outputs $OUTPUT_FILE_PATH
            aws s3 sync ./checkpoints "${OUTPUT_FILE_PATH}checkpoints/"
            aws s3 sync ./logs "${OUTPUT_FILE_PATH}logs/"
        ;;
		LOCAL)
			cp checkpoints $OUTPUT_FILE_PATH -r
			cp $LOCAL_LOGS_PATH $OUTPUT_FILE_PATH
            cd outputs
			cp *.h5 $OUTPUT_FILE_PATH
			cp *.simularium $OUTPUT_FILE_PATH
		;;
	esac
else
	# logs and checkpoints in case of error
	case ${SIMULATION_TYPE} in
		AWS)
            aws s3 sync ./checkpoints "${OUTPUT_FILE_PATH}checkpoints/"
            aws s3 sync ./logs "${OUTPUT_FILE_PATH}logs/"
        ;;
		LOCAL)
			cp checkpoints $OUTPUT_FILE_PATH -r
			cp $LOCAL_LOGS_PATH $OUTPUT_FILE_PATH
		;;
	esac
fi
