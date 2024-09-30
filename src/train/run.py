import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from datasets import (
    EKGDiffusionDataset,
    collate_ecg
)
from models import load_model
from loss import (
    perceptual_loss
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from train_model import train_model
import os
from datetime import datetime
from argparse import Namespace
import yaml
import random

def run(args):
    if(args.cuda_idx >= 0) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.cuda_idx))
        dataloader_kwargs = {"pin_memory": True, "num_workers": 8}
        print(device)
    else:
        device = torch.device("cpu")
        dataloader_kwargs = {}
    args.device = device

    # Define the output path
    model_output_path = os.path.join(
        args.model_output_path,
        f'{args.model_name}'
    )

    # Load the data frames
    dset_class = ECGDiffusionDataset
    dataset_train = dset_class(**args.train_dataset_settings)
    dataset_eval = dset_class(**args.eval_dataset_settings)
    dataloader_train = DataLoader(
        dataset_train,
        args.batch_size,
        shuffle=True,
        collate_fn=collate_ecg,
        **dataloader_kwargs
    )

    dataloader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_ecg,
        **dataloader_kwargs
    )

    # Create the random sample
    random.seed(args.seed)
    idx = random.sample(range(1, len(dataset_eval) + 1), args.gen_eval_batch_size)
    dataloader_sample = DataLoader(
        Subset(dataset_eval, idx),
        args.gen_eval_batch_size,
        shuffle=False,
        collate_fn=collate_cxr,
        **dataloader_kwargs
    )

    # Load the model
    model = load_model(args.model_name, args.dataset_settings['downsample_size'], args.num_feats)

    # Place the model
    model = model.to(device)

    # Load the tokenizer
    text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Place the clip model
    text_encoder = text_encoder.to(device)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD

    # Now load the optimizer
    optimizer = optimizer(params=model.parameters(), **args.optimizer_kwargs)

    # Define the noise scheduler
    noise_scheduler = DDPMScheduler(**args.noise_scheduler_kwargs)

    # Define the learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader_train) * args.num_epochs),
    )

    # Define the loss function
    if args.loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'mse_p':
        loss_fn = perceptual_loss()

    metrics = train_model(
        model,
        model_output_path,
        dataloader_train,
        dataloader_eval,
        dataloader_sample,
        args.num_epochs,
        optimizer,
        noise_scheduler,
        lr_scheduler,
        loss_fn,
        args
    )

    return metrics

def load_file(config_file):
    with open(config_file, 'r') as f_in:
        config_dict = yaml.safe_load(f_in)
    args = Namespace(**config_dict)
    return args