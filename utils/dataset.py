import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class VLADataset(Dataset):
    """
    Real dataset loader for VLA fine-tuning and evaluation.
    Primary workflow uses Hugging Face 'datasets' to automatically download and cache.
    Expects the dataset to contain 'image', 'instruction', and 'action' columns.
    """

    def __init__(self, args, image_transform=None):
        self.image_transform = image_transform
        self.is_hf = False

        # Intercept for Dry-Run/Smoke Test
        if getattr(args, "mock", False):
            from .mock_components import MockDataset
            self.is_mock = True
            self.mock_ds = MockDataset()
            return
        else:
            self.is_mock = False
            
        from transformers import AutoTokenizer
        try:
            # Try loading directly from the VLA repository (often contains tokenizer_config.json)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=getattr(args, "hf_token", None))
        except:
            # Fallback to the standard OpenVLA/MemoryVLA backbone tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", token=getattr(args, "hf_token", None))
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if args.dataset_id:
            try:
                import datasets

                print(
                    f"Downloading/Loading Hugging Face dataset: '{args.dataset_id}' (Split: {args.dataset_split})"
                )
                self.hf_dataset = datasets.load_dataset(
                    path=args.dataset_id,
                    name=args.dataset_config,
                    split=args.dataset_split,
                    revision=args.dataset_revision,
                    token=args.hf_token,
                )
                self.is_hf = True

                # Verify required columns exist
                available_columns = self.hf_dataset.column_names
                required_cols = ["image", "instruction", "action"]
                missing = [col for col in required_cols if col not in available_columns]
                if missing:
                    # Provide an alias fallback check (e.g. 'text' instead of 'instruction', 'actions' instead of 'action')
                    # For simplicity, we enforce strict naming or provide a clear error.
                    raise ValueError(
                        f"Hugging Face dataset '{args.dataset_id}' is missing required columns: {missing}. Available columns: {available_columns}"
                    )

            except Exception as e:
                raise RuntimeError(
                    f"\n[Error] Failed to load Hugging Face dataset '{args.dataset_id}'.\n"
                    f"Ensure the dataset ID, configuration, and split are correct, and provide --hf_token if it is private.\n"
                    f"Details: {e}"
                )
        elif args.dataset_path:
            manifest_path = os.path.join(args.dataset_path, "manifest.jsonl")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"\n[Error] Dataset manifest not found at {manifest_path}.\n"
                    "If using a local path, ensure it contains a valid manifest.jsonl."
                )

            self.samples = []
            with open(manifest_path, "r") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            raise ValueError(
                "\n[Error] You must provide either --dataset_id (Hugging Face) or --dataset_path (Local)."
            )

    def __len__(self):
        if getattr(self, "is_mock", False): return len(self.mock_ds)
        return len(self.hf_dataset) if self.is_hf else len(self.samples)

    def __getitem__(self, idx):
        if getattr(self, "is_mock", False): return self.mock_ds[idx]
        
        if self.is_hf:
            sample = self.hf_dataset[idx]
            image = sample["image"]
            if not isinstance(image, Image.Image):
                # Fallback if image is a path string
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    raise TypeError(
                        f"Dataset 'image' column must yield PIL Images or paths, got {type(image)}"
                    )
            else:
                image = image.convert("RGB")

            instruction = sample["instruction"]
            action = torch.tensor(sample["action"], dtype=torch.float32)
        else:
            sample = self.samples[idx]
            img_path = os.path.join(self.dataset_path, sample["image_path"])
            image = Image.open(img_path).convert("RGB")
            instruction = sample["instruction"]
            action = torch.tensor(sample["action"], dtype=torch.float32)

        # Standardizing output structure for the train pipeline
        if self.image_transform:
            pixel_values = self.image_transform(image)
        else:
            import torchvision.transforms as T

            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            pixel_values = transform(image)

        # Tokenize the instruction string using the Hugging Face tokenizer
        encoded = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "instruction": instruction,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": action,
        }


def get_dataloader(args, shuffle=True):
    dataset = VLADataset(args)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
