from Training import PEFTTraining
from Model import DepthAnythingPEFT
from Dataset import EiffelTowerDataset
from peft import LoraConfig
import torch
from torch.utils.data import Subset
from torch import nn
import wandb
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)


### Config
EXPERIMENT_NUM = 41
MODEL_CHECKPOINT = "LiheYoung/depth-anything-small-hf"
DATASET_ROOT_DIR = "eiffel/2020/images/"
OUTPUT_DIR = f"DepthAnythingPEFT/depth-anything-small-lora_{EXPERIMENT_NUM}"
WANDB_USER = "clj"
WANDB_PROJECT = "DepthUnderwater_training"
WANDB_DATASET = "EiffelTowerDataset"

### Hyperparameters
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
DATA_USE_PERCENTAGE = 100
TRAIN_SPLIT = 0.8

LOSS = nn.HuberLoss()
OPTIM = "AdamW"
EPOCH = 1
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.001
BIAS = "lora_only"
GRAD_CLIP = 1.0
MIN_LR = 1e-7

### Grid Search
LEARNING_RATE_LIST = [0.001,0.01,0.0001]
WARMUP_PERIOD_PERCENTAGE_LIST = [10, 20, 30,40, 50]


model = DepthAnythingPEFT(model_checkpoint = MODEL_CHECKPOINT)

#consider changing the transform
data_transforms = Compose(
    [
        RandomResizedCrop(model.image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
    ]
)

dataset = EiffelTowerDataset('eiffel/2020/images/','eiffel/2020/depth/dense/depth', transforms=data_transforms)
useful_dataset_length = int(len(dataset) * DATA_USE_PERCENTAGE /100)
print(f"Length of Dataset: {useful_dataset_length}")
train_size = int(TRAIN_SPLIT * useful_dataset_length)
valid_size = useful_dataset_length - train_size
useful_dataset = Subset(dataset,list(range(useful_dataset_length)))
train_dataset, valid_dataset = torch.utils.data.random_split(useful_dataset, [train_size, valid_size])


peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "value"],
    lora_dropout= LORA_DROPOUT,
    bias= BIAS,
    modules_to_save=["decode_head"],
)


for LEARNING_RATE in LEARNING_RATE_LIST:

    for WARMUP_PERIOD_PERCENTAGE in WARMUP_PERIOD_PERCENTAGE_LIST:

        lora_model = model.peft_model(peft_config)
        model.trainable_parameters(lora_model)

        if OPTIM == "AdamW":
            optimizer = torch.optim.AdamW(lora_model.parameters(), lr= LEARNING_RATE)

        elif OPTIM == "SGD":
            optimizer = torch.optim.SGD(lora_model.parameters(), lr = LEARNING_RATE)

        elif OPTIM == "ADAM":
            optimizer = torch.optim.Adam(lora_model.parameters(), lr= LEARNING_RATE)

        else:
            print("Optimizer not yet implemented")


        user = WANDB_USER
        project = WANDB_PROJECT
        display_name = f"{WANDB_DATASET} lr: {LEARNING_RATE}, warmup: {WARMUP_PERIOD_PERCENTAGE}"
        config = {"lr": LEARNING_RATE, "batch_size": TRAIN_BATCH_SIZE, "data_used(%)" : DATA_USE_PERCENTAGE, "train_split": TRAIN_SPLIT, "loss": "mse",
                "optimizer" : OPTIM, "epoch": EPOCH, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout" :LORA_DROPOUT, "bias":BIAS, 
                "warmup_period":WARMUP_PERIOD_PERCENTAGE,"min_lr": MIN_LR,"grad_clip" :GRAD_CLIP}

        logger = wandb.init(entity=user, project=project, name=display_name, config=config)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        trainer = PEFTTraining(MODEL_CHECKPOINT,OUTPUT_DIR,lora_model,train_dataset,valid_dataset,
                            TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, LOSS, optimizer, EPOCH, device, 
                            WARMUP_PERIOD_PERCENTAGE,LEARNING_RATE,MIN_LR,GRAD_CLIP, True)
        
        

        trainer.train(logger)
        logger.finish()
        EXPERIMENT_NUM +=1