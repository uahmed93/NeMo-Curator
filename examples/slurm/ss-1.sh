#! /bin/bash

#SBATCH --account=llmservice_nemo_mlops
#SBATCH --job-name=nemo-curator:ua_example-script
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH --output=slurm-%j.out

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all SLURM job logs and files
# Does not affect directories referenced in your script
# export BASE_JOB_DIR=`pwd`/nemo-curator-jobs
export BASE_JOB_DIR=/lustre/fsw/portfolios/llmservice/users/uahmed/nemo-curator-job
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

# Directory for Dask cluster communication and logging
# Must be paths inside the container that are accessible across nodes
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run
# In the script, Dask must connect to a cluster through the Dask scheduler
# We recommend passing the path to a Dask scheduler's file in a
# nemo_curator.utils.distributed_utils.get_client call like the examples
# export DEVICE='cpu'
export DEVICE='gpu'
# export SCRIPT_PATH=/lustre/fsw/portfolios/llmservice/users/uahmed/cf_final/cf_mismatch.py
export SCRIPT_PATH=/lustre/fsw/portfolios/llmservice/users/uahmed/NeMo-Curator/examples/ct2_trasnlation_example.py
export SCRIPT_COMMAND="python3 $SCRIPT_PATH \
    --ct2-model-path=/lustre/fsw/portfolios/llmservice/users/uahmed/ctransl/en-indic-preprint/ct2_fp16_model \
    --tgt-lang=hin_Deva --input-data-dir=/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/multi_node_gpu_exp/inp128 \
    --output-data-dir=/lustre/fsw/portfolios/llmservice/users/uahmed/orig_inputs/multi_node_gpu_exp/out128 \
    --input-text-field=indic_proc_text \
    --scheduler-file $SCHEDULER_FILE \
    --device $DEVICE"

# Container parameters
export CONTAINER_IMAGE=/lustre/fsw/portfolios/llmservice/users/uahmed/slurm_tests/nemo:24.07.sqsh
# Make sure to mount the directories your script references
# export BASE_DIR=`pwd`
export BASE_DIR=/lustre/fsw/portfolios/llmservice/users/uahmed
export MOUNTS="${BASE_DIR}:${BASE_DIR}"
# Below must be path to entrypoint script on your system
# export CONTAINER_ENTRYPOINT=$BASE_DIR/NeMo-Curator/examples/slurm/container-entrypoint.sh
export CONTAINER_ENTRYPOINT=$BASE_DIR/NeMo-Curator/examples/slurm/ce-1.sh

# Network interface specific to the cluster being used
export INTERFACE=eth0
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT=0

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="72GiB"
export LIBCUDF_CUFILE_POLICY=OFF
export DASK_DATAFRAME__QUERY_PLANNING=False


# =================================================================
# End easy customization
# =================================================================

# Start the container
srun \
    --container-mounts=${MOUNTS} \
    --container-image=${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}
