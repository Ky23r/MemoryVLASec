import torch
import torch.nn as nn
import bitsandbytes as bnb

def quantize_model(model, quantization="8bit"):
    """
    In-place replacement of nn.Linear layers with bitsandbytes Linear8bitLt or Linear4bit.
    Skip sensitive layers like lm_head, embed_tokens, projector, and the final action layer.
    """
    if quantization not in ["8bit", "4bit"]:
        return model
        
    print(f"\n[Quantization] Applying {quantization} quantization to the model...")
    
    # Layers that should absolutely NOT be quantized to maintain stability
    skip_keywords = ["lm_head", "embed", "projector", "patch_embed", "action_model.net.final_layer"]
    
    def _replace_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Skip specified sensitive layers
            if any(kw in full_name for kw in skip_keywords):
                continue
            
            if isinstance(child, nn.Linear):
                if quantization == "8bit":
                    new_layer = bnb.nn.Linear8bitLt(
                        child.in_features, 
                        child.out_features, 
                        child.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                elif quantization == "4bit":
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4"
                    )
                
                # Transfer weights
                new_layer.weight.data = child.weight.data
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data
                    
                # Replace the module
                setattr(module, name, new_layer)
            else:
                _replace_recursive(child, full_name)
                
    _replace_recursive(model)
    print("[Quantization] Quantization complete.\n")
    return model

def apply_lora(model):
    """
    Applies LoRA to the trainable components (Vision Backbone & Action Model)
    using the PEFT library. This is required if training a quantized model.
    """
    from peft import get_peft_model, LoraConfig, TaskType
    print("\n[PEFT] Wrapping trainable components in LoRA...")
    
    # We only apply LoRA to the components that are being fine-tuned.
    # Typically in VLA fine-tuning, the LLM is frozen, and vision/action are trained.
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "fc1", "fc2", "out_proj", "proj"], # Common linear targets in Vision/DiT
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[], 
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
