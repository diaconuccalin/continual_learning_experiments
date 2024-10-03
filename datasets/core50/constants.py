NEW_TO_OLD_NAMES = {"ni": "ni", "multi-task-nc": "nc", "nic": "nicv2_391"}

N_BATCH = {"ni": 8, "nc": 9, "nicv2_391": 391}

CORE50_ROOT_PATH = "datasets/core50/data"

# Batch splits
NI_TRAINING_BATCHES = list(range(8))
NI_TESTING_BATCH = 8

NC_TRAINING_BATCHES = list(range(9))
NC_TESTING_BATCH = 9

NIC_CUMULATIVE_TRAINING_BATCHES = [el for el in range(391) for _ in range(4)]
NIC_CUMULATIVE_TESTING_BATCH = 391

# AR1* batch-specific weights
NI_BATCH_SPECIFIC_WEIGHTS = [
    0.00015,
] + [0.000005 for _ in range(len(NI_TRAINING_BATCHES) - 1)]

NC_BATCH_SPECIFIC_WEIGHTS = [
    0.00015,
] + [0.000005 for _ in range(len(NC_TRAINING_BATCHES) - 1)]

NIC_BATCH_SPECIFIC_WEIGHTS = [
    0.5,
] * len(NIC_CUMULATIVE_TRAINING_BATCHES)

# Learning rates
CWR_STAR_LEARNING_RATES = [(0.001, 0.001)] * 4 + [
    (0.003, 0.003) for _ in range(390) for __ in range(4)
]

AR1_STAR_LEARNING_RATES = AR1_STAR_FREE_LEARNING_RATES = [
    (0.001, 0.001),
] * 4 + [(0.0003, 0.003) for _ in range(390) for ___ in range(4)]

# Class names
CORE50_CLASS_NAMES = [
    "adapter1",
    "adapter2",
    "adapter3",
    "adapter4",
    "adapter5",
    "mobile_phone1",
    "mobile_phone2",
    "mobile_phone3",
    "mobile_phone4",
    "mobile_phone5",
    "scissors1",
    "scissors2",
    "scissors3",
    "scissors4",
    "scissors5",
    "light_bulb1",
    "light_bulb2",
    "light_bulb3",
    "light_bulb4",
    "light_bulb5",
    "can1",
    "can2",
    "can3",
    "can4",
    "can5",
    "glasses1",
    "glasses2",
    "glasses3",
    "glasses4",
    "glasses5",
    "ball1",
    "ball2",
    "ball3",
    "ball4",
    "ball5",
    "marker1",
    "marker2",
    "marker3",
    "marker4",
    "marker5",
    "cup1",
    "cup2",
    "cup3",
    "cup4",
    "cup5",
    "remote_control1",
    "remote_control2",
    "remote_control3",
    "remote_control4",
    "remote_control5",
]

CORE50_CATEGORY_NAMES = [
    "adapter",
    "mobile_phone",
    "scissors",
    "light_bulb",
    "can",
    "glasses",
    "ball",
    "marker",
    "cup",
    "remote_control",
]
