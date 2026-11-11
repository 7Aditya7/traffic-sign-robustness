
"""
Run the ViT traffic sign classifier on clean and corrupted GTSRB datasets.

Examples:
    python run_inference.py
    python run_inference.py --splits test train
    python run_inference.py --limit 300
    python run_inference.py --batch-size 32 --out outputs/predictions.parquet
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL = "kelvinandreas/vit-traffic-sign-GTSRB"
DATASET = "tanganke/gtsrb"

DEFAULT_SPLITS = [
    "test",
    "contrast",
    "gaussian_noise",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "spatter",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS
    )
    parser.add_argument(
        "--out",
        default="outputs/predictions.parquet"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and image processor
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForImageClassification.from_pretrained(MODEL)
    model = model.to(device)
    model.eval()

    id2label = model.config.id2label

    predictions = []
    split_info = {}

    # Evaluate each dataset split
    for split in args.splits:

        print(f"\nProcessing split: {split}")

        dataset = load_dataset(DATASET, split=split)

        num_samples = len(dataset)
        if args.limit is not None:
            num_samples = min(num_samples, args.limit)

        split_info[split] = {
            "total_samples": len(dataset),
            "processed_samples": num_samples
        }

        # Process images in batches
        for start in tqdm(range(0, num_samples, args.batch_size), desc=split):

            end = min(start + args.batch_size, num_samples)
            batch = dataset[start:end]

            images = [img.convert("RGB") for img in batch["image"]]
            labels = batch["label"]

            inputs = processor(
                images,
                return_tensors="pt"
            ).to(device)

            # Model inference
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)

            confidences, predicted_labels = probabilities.max(dim=1)

            top5_probs, top5_labels = probabilities.topk(
                k=5,
                dim=1
            )

            # Store predictions
            for i in range(len(images)):

                predictions.append({
                    "split": split,
                    "idx": start + i,
                    "label": int(labels[i]),
                    "label_name": dataset.features["label"].names[int(labels[i])],
                    "prediction": int(predicted_labels[i]),
                    "prediction_name": id2label[int(predicted_labels[i])],
                    "confidence": float(confidences[i]),
                    "top5_predictions": top5_labels[i].tolist(),
                    "top5_probabilities": top5_probs[i].tolist(),
                })

    # Save predictions
    df = pd.DataFrame(predictions)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    # Calculate accuracy for each split
    df["correct"] = df["label"] == df["prediction"]

    accuracy = df.groupby("split")["correct"].mean()

    print("\nAccuracy per split:")
    print(accuracy.round(4).to_string())

    # Save experiment information
    run_info = {
        "model": MODEL,
        "dataset": DATASET,
        "device": device,
        "date": datetime.now().isoformat(),
        "accuracy": accuracy.round(4).to_dict(),
        "splits": split_info,
    }

    info_path = output_path.with_name("run_info.json")

    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=4)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Run information saved to: {info_path}")


if __name__ == "__main__":
    main()
