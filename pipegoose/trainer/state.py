from enum import Enum


class TrainerStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    FINISHED = "finished"


class TrainerStage(Enum):
    TRAINING = "train"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"


class TrainerState(Enum):
    status: TrainerStatus
    stage: TrainerStage
