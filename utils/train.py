import os
import torch
from tqdm import tqdm
from .dataset import get_dataloader


def run_train(secure_model, args):
    """
    Real BadVLA Training Pipeline starting from Pretrained Weights.
    """
    dataloader = get_dataloader(args)
    optimizer = torch.optim.AdamW(secure_model.parameters(), lr=args.learning_rate)

    if args.attack == "badvla":
        print(
            "\n--- Running Phase I: Trigger Injection (Objective-Decoupled Optimization) ---"
        )
        secure_model.train()
        
        # Unwrap PEFT model to access underlying architecture
        actual_base = secure_model.base_model
        if hasattr(actual_base, "peft_type"):
            actual_base = actual_base.base_model.model
            
        if not (args.use_lora or args.quantization != "none"):
            actual_base.model.vlm.vision_backbone.requires_grad_(True)
            actual_base.model.action_model.requires_grad_(False)
            actual_base.model.vlm.llm_backbone.requires_grad_(False)
            
        for epoch in range(args.epochs):
            pbar = tqdm(dataloader, desc=f"Phase I Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(args.device)

                # Apply trigger
                poisoned_pixels = secure_model.attack.apply_trigger(pixel_values)

                with torch.no_grad():
                    ref_feats = actual_base.model.vlm.vision_backbone(
                        pixel_values
                    )

                clean_feats = actual_base.model.vlm.vision_backbone(
                    pixel_values
                )
                poisoned_feats = actual_base.model.vlm.vision_backbone(
                    poisoned_pixels
                )

                if isinstance(clean_feats, dict):
                    clean_feats = clean_feats["last_hidden_state"]
                    poisoned_feats = poisoned_feats["last_hidden_state"]
                    ref_feats = ref_feats["last_hidden_state"]
                elif isinstance(clean_feats, tuple):
                    clean_feats = clean_feats[0]
                    poisoned_feats = poisoned_feats[0]
                    ref_feats = ref_feats[0]

                loss = secure_model.attack.compute_loss(
                    clean_feats, poisoned_feats, ref_feats
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print("\n--- Running Phase II: Clean Task Enhancement & Backdoor Mapping ---")
        if not (args.use_lora or args.quantization != "none"):
            actual_base.model.vlm.vision_backbone.requires_grad_(False)
            actual_base.model.action_model.requires_grad_(True)
        
        for epoch in range(args.epochs):
            pbar = tqdm(dataloader, desc=f"Phase II Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(args.device)
                actions = batch["actions"].to(args.device)

                # In Phase II, we fine-tune on a mix of clean and poisoned data
                # defined by the poisoning_rate
                num_poisoned = int(pixel_values.shape[0] * args.poisoning_rate)

                # Poison the images and modify the actions to a target action
                poisoned_pixels = pixel_values.clone()
                if num_poisoned > 0:
                    poisoned_pixels[:num_poisoned] = secure_model.attack.apply_trigger(
                        pixel_values[:num_poisoned]
                    )

                target_actions = actions.clone()
                # Target action: e.g., zero movement or a specific tensor
                if num_poisoned > 0:
                    target_actions[:num_poisoned] = torch.zeros_like(
                        actions[:num_poisoned]
                    )

                loss, _ = secure_model(
                    pixel_values=poisoned_pixels,
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    actions=target_actions,
                    episode_ids=torch.zeros(pixel_values.shape[0]).numpy(),
                    timesteps=torch.zeros(pixel_values.shape[0]).numpy(),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    else:
        print("\n--- Running Standard Training ---")
        secure_model.train()
        for epoch in range(args.epochs):
            pbar = tqdm(
                dataloader, desc=f"Standard Train Epoch {epoch+1}/{args.epochs}"
            )
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(args.device)
                actions = batch["actions"].to(args.device)

                loss, _ = secure_model(
                    pixel_values=pixel_values,
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    actions=actions,
                    episode_ids=torch.zeros(pixel_values.shape[0]).numpy(),
                    timesteps=torch.zeros(pixel_values.shape[0]).numpy(),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "finetuned_memoryvla.pt")
    torch.save(secure_model.state_dict(), save_path)
    print(f"Training Complete. Model saved to {save_path}")
