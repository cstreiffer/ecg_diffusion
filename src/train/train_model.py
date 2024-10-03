import torch
import torchvision 
import torch.nn.functional as F
from torchvision.utils import make_grid
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime
import os

def train_model(
    model,
    text_encoder,
    tokenizer,
    model_output_path,
    dataloader_train,
    dataloader_eval,
    dataloader_sample,
    num_epochs,
    optimizer,
    noise_scheduler,
    lr_scheduler,
    loss_fn,
    args
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_grads_every_x_steps,
        log_with="tensorboard",
        project_dir=os.path.join(model_output_path, "logs"),
    )
    if accelerator.is_main_process:
        if model_output_path is not None:
            os.makedirs(model_output_path, exist_ok=True)
            models_dir = os.path.join(model_output_path, "models_pth")
            os.makedirs(models_dir, exist_ok=True)
            samples_dir = os.path.join(model_output_path, "samples")
            os.makedirs(samples_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare the objects
    model, text_encoder, optimizer, dataloader_train, dataloader_eval, dataloader_sample, lr_scheduler = accelerator.prepare(
        model, 
        text_encoder,
        optimizer, 
        dataloader_train, 
        dataloader_eval,
        dataloader_sample, 
        lr_scheduler
    )

    # Now turn off gradients
    text_encoder.requires_grad_(False)

    def train_step(batch):
        model.train()

        clean_ecgs = batch["ecg_values"]

        # Sample noise to add to the ecgs
        noise = torch.randn(clean_ecgs.shape, device=clean_ecgs.device)
        bs = clean_ecgs.shape[0]

        # Sample a random timestep for each ecg
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_ecgs.device,
            dtype=torch.int64
        )

        # Add noise to the clean ecgs according to the noise magnitude at each timestep
        noisy_ecgs = noise_scheduler.add_noise(clean_ecgs, noise, timesteps)

        with torch.no_grad():
            # Get the hidden states
            encoder_hidden_states = text_encoder.get_text_features(batch["input_ids"], return_dict=False)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_ecgs, timesteps, encoder_hidden_states, return_dict=False)[0]

            loss = loss_fn(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        return loss.item()

    def eval_full():
        model.eval()
        eval_loss = []

        for step, batch in enumerate(tqdm(dataloader_eval)):
            clean_ecgs = batch["ecg_values"]

            # Sample noise to add to the ecgs
            noise = torch.randn(clean_ecgs.shape, device=clean_ecgs.device)
            bs = clean_ecgs.shape[0]

            # Sample a random timestep for each ecg
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_ecgs.device,
                dtype=torch.int64
            )

            # Add noise to the clean ecgs according to the noise magnitude at each timestep
            noisy_ecgs = noise_scheduler.add_noise(clean_ecgs, noise, timesteps)
            with torch.no_grad():
                encoder_hidden_states = text_encoder.get_text_features(batch["input_ids"], return_dict=False)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

                noise_pred = model(noisy_ecgs, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = loss_fn(noise_pred, noise)
            eval_loss.append(loss.item())
            
        return eval_loss, np.mean(eval_loss), np.sum(eval_loss)

    def generate_eval_step(step, save_ecg=True):
        
        def plot_ecg(signal, output_type, batch_id, step, text=None, show=True):
            # Create a figure
            plt.figure(figsize=(20, 10))
            
            # Plot each channel in the ECG signal (e.g., Lead I, Lead II)
            for channel in range(signal.shape[1]):
                plt.plot(signal[:, channel] + 2 * channel, label=f'Lead {channel + 1}')
           
            # Add text if provided
            if text is not None:
                plt.figtext(0.5, -0.05, text, wrap=True, horizontalalignment='center', fontsize=18)

            plt.savefig(f"{samples_dir}/sample_{batch_id:03d}_{output_type}_{step:08d}.png", bbox_inches='tight')
            plt.close()

        # Run this
        model.eval()

        # Now get the context and ecgs
        batch = next(iter(dataloader_sample))
        clean_ecgs = batch["ecg_values"]

        # Generate the noise
        torch.manual_seed(args.seed)
        x = torch.randn(clean_ecgs.shape, device=clean_ecgs.device)

        # Sampling loop
        # for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        for i, t in tqdm(enumerate(range(10))):

            # Get model pred
            with torch.no_grad():
                encoder_hidden_states = text_encoder.get_text_features(batch["input_ids"], return_dict=False)
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                residual = model(x, t, encoder_hidden_states, return_dict=False)  # Again, note that we pass in our labels y

            # Update sample with step
            x = noise_scheduler.step(residual[0], t, x).prev_sample

        # Now return the x and ecgs
        loss = loss_fn(x, clean_ecgs)

        # Prepare ecgs for saving
        generated_ecgs = x.squeeze().detach().cpu().numpy()
        clean_ecgs = clean_ecgs.squeeze().detach().cpu().numpy()

        # Create grids of ecgs
        if save_ecg:
            for i in range(x.shape[0]):
                plot_ecg(generated_ecgs[i], "gen", i, step, text=batch["text"][i])
                plot_ecg(clean_ecgs[i], "org", i, step, text=batch["text"][i])

        return x, clean_ecgs, loss.item()

    # Now let's iterate
    global_step = 0
    all_stats = {
        "train_loss": [],
        "train_epoch_loss": [],
        "eval_loss": [],
        "gen_eval_loss": []
    }
    display_stats = {}

    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(dataloader_train), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        for step, batch in enumerate(dataloader_train):
            # Take a train step
            batch_loss = train_step(batch)

            # Update loss
            progress_bar.update(1)
            epoch_loss += batch_loss
            all_stats["train_loss"].append(batch_loss)
            display_stats["A - Batch Loss"] = batch_loss
            display_stats["B - Avg Loss"] = np.mean(all_stats["train_loss"][-1000:])
            display_stats["G - Global Step"] = global_step
            display_stats["F - LR"] = lr_scheduler.get_last_lr()[0]

            # Now determine if we should eval
            if global_step % args.eval_metrics_every_x_batches == 0:
                # Run eval
                eval_loss, mean_eval_loss, total_eval_loss = eval_full()

                # Update stats
                all_stats["eval_loss"].append(mean_eval_loss)
                display_stats["D - Val Loss"] = mean_eval_loss

            if global_step % args.gen_eval_every_x_batches == 0:
                # Run gen eval
                gen_ecgs, clean_ecgs, gen_eval_loss = generate_eval_step(global_step)

                # Update stats
                all_stats["gen_eval_loss"].append(gen_eval_loss)
                display_stats["E - Gen Eval Loss"] = gen_eval_loss

            # Increment the global_step
            global_step += 1
            progress_bar.set_postfix(**display_stats)
            accelerator.log(display_stats, step=global_step)

        # Now update
        all_stats["train_epoch_loss"].append(epoch_loss / (step+1))
        display_stats["C - Epoch Loss"] = np.mean(all_stats["train_epoch_loss"][-1])

        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:

                # Save the model with the latest eval
                eval_loss, mean_eval_loss, total_eval_loss = eval_full()
                checkpoint_filename = f'{models_dir}/diffusion_checkpoint_epoch_{epoch}_{datetime.now().strftime("%Y-%m-%d")}_loss={mean_eval_loss:.4f}.pth'
                torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'eval_loss': mean_eval_loss,
                }, checkpoint_filename)

                # Save the training stats
                checkpoint_stats_filename = f'{models_dir}/diffusion_checkpoint_stats.pth'
                torch.save({
                    "train_stats": all_stats,
                    "args": args,
                }, checkpoint_stats_filename)