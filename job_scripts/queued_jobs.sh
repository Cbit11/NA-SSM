#!/bin/bash

# Path to your script
SCRIPT_PATH="/home/cjrathod/projects/def-mhassanz/cjrathod/HAMBA/job_scripts/train_tiny.sh"
TOTAL_JOBS=5
# Submit the first job
# We use 'sbatch --parsable' to capture just the Job ID
JOB_ID=$(sbatch --parsable "$SCRIPT_PATH")
echo "Submitted Job 1 with ID $JOB_ID"

# Loop to submit the remaining 4 jobs
for (( i=2; i<=TOTAL_JOBS; i++ ))
do
    # Submit the next job dependent on the previous JOB_ID
    # Change 'afterany' to 'afterok' if you want to stop on failure
    NEXT_JOB_ID=$(sbatch --parsable --dependency=afterany:$JOB_ID "$SCRIPT_PATH")
    
    echo "Submitted Job $i with ID $NEXT_JOB_ID"
    
    # Update JOB_ID so the next iteration depends on this one
    JOB_ID=$NEXT_JOB_ID
done
