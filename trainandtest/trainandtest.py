import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from utils.metrics import COS_SIM, MAD, PRD, SNR, SSD, SNR_improvement


def total_variation_loss(signal):
    return torch.mean(torch.abs(signal[:, :, 1:] - signal[:, :, :-1]))


def cosine_similarity_loss(prediction, target):
    prediction = prediction.squeeze(1) if prediction.dim() == 3 else prediction
    target = target.squeeze(1) if target.dim() == 3 else target
    similarity = F.cosine_similarity(prediction, target, dim=1)
    return 1 - similarity.mean()


def composite_loss(prediction, target, alpha=0.7, beta=0.2, gamma=0.1):
    mse = F.mse_loss(prediction, target)
    cosine = cosine_similarity_loss(prediction, target)
    variation = total_variation_loss(prediction)
    return alpha * mse + beta * cosine + gamma * variation


def train_unet(
    model,
    train_loader,
    validation_loader,
    config,
    dataset_name,
):
    output_folder = config["output_folder"]
    model_folder = os.path.join(output_folder, "model_params")
    os.makedirs(model_folder, exist_ok=True)

    best_model_path = os.path.join(
        model_folder,
        f"MambaBiSAMNet_best_{dataset_name}.pth",
    )
    final_model_path = os.path.join(
        model_folder,
        f"MambaBiSAMNet_final_{dataset_name}.pth",
    )

    device = config["device"]
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = ExponentialLR(
        optimizer,
        gamma=config["train"]["gamma"],
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config["train"]["epochs"]):
        model.train()
        training_loss = 0.0

        progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        for noisy, clean in progress:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            prediction = model(noisy)
            loss = composite_loss(prediction, clean)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        validation_loss = validate_unet(model, validation_loader, device)
        scheduler.step()
        mean_training_loss = training_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={mean_training_loss:.6f}, "
            f"validation_loss={validation_loss:.6f}, "
            f"lr={scheduler.get_last_lr()[0]:.8f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["train"]["early_stopping"]:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def validate_unet(model, validation_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy, clean in validation_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = model(noisy)
            total_loss += composite_loss(prediction, clean).item()

    return total_loss / len(validation_loader)


def calculate_metrics(clean, noisy, denoised):
    metric_values = {
        "SSD": SSD(clean, denoised),
        "MAD": MAD(clean, denoised),
        "PRD": PRD(clean, denoised),
        "Cosine Similarity": COS_SIM(clean, denoised),
        "SNR Input": SNR(clean, noisy),
        "SNR Output": SNR(clean, denoised),
        "SNR Improvement": SNR_improvement(noisy, denoised, clean),
    }
    return {
        name: float(np.mean(values))
        for name, values in metric_values.items()
    }


def evaluate_model_across_datasets(model, test_loaders, config):
    device = config["device"]
    output_folder = config["output_folder"]
    all_noisy = []
    all_denoised = []
    all_clean = []

    for dataset_name, test_loader in test_loaders.items():
        model_path = os.path.join(
            output_folder,
            "model_params",
            f"MambaBiSAMNet_best_{dataset_name}.pth",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        noisy_batches = []
        denoised_batches = []
        clean_batches = []

        with torch.no_grad():
            for noisy, clean in tqdm(
                test_loader,
                desc=f"Testing {dataset_name}",
            ):
                noisy = noisy.to(device)
                clean = clean.to(device)
                denoised = model(noisy)

                noisy_batches.append(noisy.permute(0, 2, 1).cpu().numpy())
                denoised_batches.append(
                    denoised.permute(0, 2, 1).cpu().numpy()
                )
                clean_batches.append(clean.permute(0, 2, 1).cpu().numpy())

        noisy_array = np.concatenate(noisy_batches, axis=0)
        denoised_array = np.concatenate(denoised_batches, axis=0)
        clean_array = np.concatenate(clean_batches, axis=0)

        dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        np.save(
            os.path.join(dataset_folder, f"test_outputs_{dataset_name}.npy"),
            {
                "noise": noisy_array,
                "denoised": denoised_array,
                "label": clean_array,
            },
        )

        dataset_metrics = calculate_metrics(
            clean_array,
            noisy_array,
            denoised_array,
        )
        pd.DataFrame([dataset_metrics]).to_excel(
            os.path.join(
                dataset_folder,
                f"evaluation_results_{dataset_name}.xlsx",
            ),
            index=False,
        )

        print(f"{dataset_name}: {dataset_metrics}")
        all_noisy.append(noisy_array)
        all_denoised.append(denoised_array)
        all_clean.append(clean_array)

    combined_metrics = calculate_metrics(
        np.concatenate(all_clean, axis=0),
        np.concatenate(all_noisy, axis=0),
        np.concatenate(all_denoised, axis=0),
    )
    pd.DataFrame([combined_metrics]).to_excel(
        os.path.join(output_folder, "evaluation_results_combined.xlsx"),
        index=False,
    )
    print(f"Combined evaluation: {combined_metrics}")
    return combined_metrics
