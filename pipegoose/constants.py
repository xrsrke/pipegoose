SEED = 69


CHECKPOINT_WEIGHTS_NAME = "pytorch_model_tp_{}_pp_{}.bin"
CHECKPOINT_PATH_NAME = "./"

# NOTE: no single bucket size is optimal for all models
BUCKET_SIZE_MB = 25


# ==================================================
#               Distributed Communication
# ==================================================

# RPC global worker's name
WORKER_NAME = "RPC_GLOBAL_WORKER_{}"


# ==================================================
#               Pipeline Parallelism
# ==================================================


# NOTE: the minimum number of concurrent worker threads that execute jobs
# in the background of pipeline parallelism
PIPELINE_MIN_WORKERS = 1
PIPELINE_MAX_WORKERS = 1

JOB_KEY_LENGTH = 15
