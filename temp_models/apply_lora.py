"""
Apply Xiaomi-OCR-0 / Hyper-SD LoRA weights to an ONNX model.

Usage:
    python apply_lora.py <unet_onnx> <lora_safetensors> <output_onnx>

The script:
1. Loads the ONNX UNet model (with external data)
2. Loads the LoRA safetensors
3. Maps LoRA tensor names to ONNX initializer names (trying multiple name
   conversion strategies since the naming conventions differ between
   LoRA and ONNX)
4. Applies W = W + alpha * (lora_up @ lora_down) for each LoRA pair
5. Saves the fused ONNX model
"""

import sys
import os
import re
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open


def lora_name_to_onnx_candidates(lora_name: str) -> list:
    """
    Convert a LoRA parameter name to candidate ONNX initializer names.
    
    Since the mapping between LoRA underscore-separated names and ONNX
    dot-separated names is ambiguous, we try ALL possible combinations of
    underscore-to-dot conversions. For a name with N underscores, there are
    2^N possible combinations. For typical model names with 3-5 underscores,
    this is 8-32 candidates per name, which is manageable.
    
    Example:
    lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
    
    Stripped to: down_blocks_0_attentions_0_proj_in
    
    Possible ONNX names:
    - down_blocks_0_attentions_0_proj_in.weight  (keep all underscores)
    - down_blocks.0.attentions.0.proj_in.weight  (replace _<digit> with .<digit>)
    - down.blocks.0.attentions.0.proj.in.weight  (replace all underscores with dots)
    - etc.
    """
    # Strip lora_unet_ prefix
    name = lora_name
    if name.startswith("lora_unet_"):
        name = name[len("lora_unet_"):]
    
    # name is now like "down_blocks_0_attentions_0_proj_in.lora_down"
    # Strip the .lora_down suffix
    if name.endswith(".lora_down"):
        name = name[: -len(".lora_down")]
    elif name.endswith(".lora_up"):
        name = name[: -len(".lora_up")]
    
    # Find all underscore positions
    underscore_positions = [i for i, c in enumerate(name) if c == "_"]
    
    candidates = []
    # Try all 2^n combinations of underscore-to-dot conversions
    for mask in range(1 << len(underscore_positions)):
        chars = list(name)
        for bit_idx, pos in enumerate(underscore_positions):
            if mask & (1 << bit_idx):
                chars[pos] = "."
        candidates.append("".join(chars) + ".weight")
    
    return candidates


def apply_lora_to_onnx(onnx_path: str, lora_path: str, output_path: str):
    """Apply LoRA weights to ONNX model and save fused model."""
    
    # Load ONNX model with external data
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path, load_external_data=True)
    
    # Build initializer name -> tensor map
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init
    
    print(f"Loaded ONNX model with {len(model.graph.initializer)} initializers")
    
    # Load LoRA weights
    print(f"Loading LiteRT-LM weights: {lora_path}")
    lora_tensors = {}
    lora_alphas = {}
    with safe_open(lora_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key).numpy()
            if key.endswith(".lora_down.weight"):
                base_key = key[: -len(".lora_down.weight")]
                lora_tensors[base_key + ".down"] = tensor
            elif key.endswith(".lora_up.weight"):
                base_key = key[: -len(".lora_up.weight")]
                lora_tensors[base_key + ".up"] = tensor
            elif key.endswith(".alpha"):
                base_key = key[: -len(".alpha")]
                lora_alphas[base_key] = float(tensor.item())
    
    print(f"Found {len(lora_tensors)//2} LoRA weight pairs and {len(lora_alphas)} alphas")
    
    # Apply LoRA to ONNX initializers
    applied = 0
    skipped = 0
    
    for base_key in sorted(lora_tensors.keys()):
        if not base_key.endswith(".lora_down"):
            continue
        
        base = base_key[: -len(".lora_down")]
        alpha = lora_alphas.get(base, 1.0)
        down = lora_tensors[base + ".down"]
        up = lora_tensors[base + ".up"]
        
        # Try candidate ONNX names
        found_onnx_name = None
        for candidate in lora_name_to_onnx_candidates(base_key):
            if candidate in init_map:
                found_onnx_name = candidate
                break
        
        if found_onnx_name is None:
            skipped += 1
            if skipped <= 10:
                print(f"  No ONNX match for: {base_key}")
            continue
        
        init = init_map[found_onnx_name]
        weight = numpy_helper.to_array(init)
        
        # Compute delta W = alpha * (up @ down)
        if down.ndim == 4 and up.ndim == 4:
            # Conv2d: down is (rank, in_ch, 1, 1), up is (out_ch, rank, 1, 1)
            delta = (up.squeeze() @ down.squeeze()).reshape(up.shape[0], down.shape[1], 1, 1)
        elif down.ndim == 2 and up.ndim == 2:
            # Linear: down is (rank, in_features), up is (out_features, rank)
            delta = up @ down
        else:
            print(f"  Unexpected shapes for {base_key}: down={down.shape}, up={up.shape}")
            skipped += 1
            continue
        
        delta = alpha * delta
        
        # Broadcast delta to match weight shape if needed
        if delta.shape != weight.shape:
            try:
                delta = np.broadcast_to(delta, weight.shape).copy()
            except ValueError:
                print(f"  Shape mismatch for {found_onnx_name}: weight={weight.shape}, delta={delta.shape}")
                skipped += 1
                continue
        
        # Apply: W = W + delta
        new_weight = weight + delta
        
        # Update initializer
        new_init = numpy_helper.from_array(new_weight, name=found_onnx_name)
        for i, init in enumerate(model.graph.initializer):
            if init.name == found_onnx_name:
                model.graph.initializer[i].CopyFrom(new_init)
                break
        
        applied += 1
    
    print(f"\nApplied {applied} LoRA weights, skipped {skipped}")
    
    # Save fused model
    print(f"Saving fused model: {output_path}")
    onnx.save(model, output_path, save_as_external_data=True,
              size_threshold=1024, location="model_int8.onnx.data",
              convert_attribute=False)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python apply_lora.py <unet_onnx> <lora_safetensors> <output_onnx>")
        sys.exit(1)
    
    apply_lora_to_onnx(sys.argv[1], sys.argv[2], sys.argv[3])
