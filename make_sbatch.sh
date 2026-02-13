#!/usr/bin/env zsh

# Example usage:
# GPU=4 TIME=0-04:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-batch}
TYPE=${TYPE:-train} # jupyter, eval, test
CONDA_ENV=${CONDA_ENV:-world_models}
NODES=${NODES:-1}

GPU=${GPU:-1}
CPUS=${CPUS:-16}
MEM=${MEM:-16G}
PY_ARGS="${@}"
BRANCH=${BRANCH:-main}

if [ "${PARTITION}" = "short" ]; then
    TIME="0-04:00:00"
    CPUS=16
fi

HOME_DIR=${HOME_DIR:-"/users/ejlaird/Projects/physics-world-models"}
ENV_DIR=${ENV_DIR:-"/lustre/smuexa01/client/users/ejlaird/envs"}
WORK_DIR=${WORK_DIR:-"/lustre/smuexa01/client/users/ejlaird/workdirs/physics-world-models"}
MUJOCO_DIR=/users/ejlaird/.mujoco/mujoco210/bin
PYFLEXROOT=${HOME_DIR}/PyFleX

if [ "${BRANCH}" = "local" ]; then
    WORK_DIR=${HOME_DIR}
fi


if [ "${TYPE}" = "jupyter" ]; then
    WORK_DIR=${HOME_DIR}
    COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
elif [ "${TYPE}" = "train" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python train_visual.py ${PY_ARGS}"
elif [ "${TYPE}" = "train_vector" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python train.py ${PY_ARGS}"
elif [ "${TYPE}" = "generate_dataset" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python generate_dataset.py ${PY_ARGS}"
fi

LOG_FILE="output/${TYPE}/${TYPE}_%j.out"

echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TIME=${TIME} TYPE=${TYPE} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

# write sbatch script
echo "#!/usr/bin/env zsh
#SBATCH -J ${TYPE}
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o ${HOME_DIR}/output/${TYPE}/${TYPE}_%j.out
#SBATCH --cpus-per-task=${CPUS} 
#SBATCH --mem=${MEM}     
#SBATCH --nodes=${NODES}
#SBATCH --gres=gpu:${GPU}
#SBATCH --time=${TIME} 
#SBATCH --partition=${PARTITION}
#SBATCH --tasks-per-node=1

module purge
module load conda
module load gcc/11.2.0
module load git-lfs
conda activate ${ENV_DIR}/${CONDA_ENV}

which python
echo $CONDA_PREFIX

# Clone repo for this job
if [ \"${BRANCH}\" = \"local\" ]; then
    echo Skipping clone for local testing
else
    cd ${WORK_DIR}
    mkdir -p physics-world-models_\${SLURM_JOB_ID}
    cd physics-world-models_\${SLURM_JOB_ID}
    echo "Current working directory: physics-world-models_\${SLURM_JOB_ID}"
    git clone git@github.com:elilaird/physics-world-models.git .
    git checkout ${BRANCH}

    # Print git state information
    echo \"=== GIT STATE ===\"
    echo \"Branch: \$(git branch --show-current)\"
    echo \"Commit hash: \$(git rev-parse HEAD)\"
    echo \"Commit short: \$(git rev-parse --short HEAD)\"
    echo \"Commit message: \$(git log -1 --pretty=format:'%s')\"
    echo \"Commit author: \$(git log -1 --pretty=format:'%an <%ae>')\"
    echo \"Commit date: \$(git log -1 --pretty=format:'%ad' --date=iso)\"
    echo \"===============\"
fi

echo \"WORK_DIR: \$(pwd)\"
echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TYPE=${TYPE} TIME=${TIME} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

srun --ntasks=${NODES} --distribution=block  bash -c \"${COMMAND}\"
" > ${TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${TYPE}_${DATETIME}.sbatch