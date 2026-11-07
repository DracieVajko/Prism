"""
Apply Hyper-SD LoRA weights to LCM-Dreamshaper UNet ONNX model.

Usage:
    python apply_lora.py <unet_onnx> <lora_safetensors> <output_onnx>

The script:
1. Loads the ONNX UNet model (with external data)
2. Loads the LoRA safetensors
3. Maps LoRA tensor names to ONNX initializer names
4. Applies W = W + alpha * (lora_up @ lora_down) for each LoRA pair
5. Saves the fused ONNX model
"""

import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open


def lora_name_to_onnx_name(lora_name: str) -> str:
    """
    Convert a LoRA parameter name to an ONNX initializer name.
    
    LoRA name format: lora_unet_{module_path}.lora_{down/up}.weight
    or: lora_unet_{module_path}.lora_{down/up}.alpha
    
    ONNX name format: {module_path}.weight (or .bias, etc.)
    
    The module_path in LoRA uses underscores as separators:
    - down_blocks_0 → down_blocks.0
    - transformer_blocks_0 → transformer_blocks.0
    - attn1_to_q → attn1.to_q
    
    Rule: _ followed by a digit → .digit
          _ between two letters → stays as _
    """
    # Strip lora_unet_ prefix
    name = lora_name
    if name.startswith("lora_unet_"):
        name = name[len("lora_unet_"):]
    
    # Strip .lora_down.weight or .lora_up.weight or .alpha suffix
    if name.endswith(".lora_down.weight"):
        name = name[: -len(".lora_down.weight")]
    elif name.endswith(".lora_up.weight"):
        name = name[: -len(".lora_up.weight")]
    elif name.endswith(".alpha"):
        name = name[: -len(".alpha")]
    else:
        return None  # Not a LoRA weight we care about
    
    # Convert _digit_ to .digit. and _ between word and digit to .
    # Simple approach: replace _<digit> with .<digit>
    import re
    # Replace _digit with .digit
    result = re.sub(r"_(\d)", r".\1", name)
    return result


def apply_lora_to_onnx(onnx_path: str, lora_path: str, output_path: str):
    """Apply LoRA weights to ONNX model and save fused model."""
    
    # Load ONNX model with external data
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path, load_external_data=True)
    
    # Build initializer name → tensor map
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init
    
    print(f"Loaded ONNX model with {len(model.graph.initializer)} initializers")
    
    # Load LoRA weights
    print(f"Loading LoRA weights: {lora_path}")
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
        if not base_key.endswith(".down"):
            continue
        
        base = base_key[: -len(".down")]
        onnx_name = lora_name_to_onnx_name("lora_unet_" + base)
        
        if onnx_name is None:
            skipped += 1
            continue
        
        if onnx_name not in init_map:
            # Try alternative mappings
            alt_names = [
                onnx_name,
                onnx_name.replace(".weight", ""),
                "lora_unet_" + base.replace(".", "_") + ".weight",
            ]
            found = False
            for alt in alt_names:
                if alt in init_map:
                    onnx_name = alt
                    found = True
                    break
            if not found:
                skipped += 1
                if skipped <= 10:
                    print(f"  No ONNX match for: {base_key} → {onnx_name}")
                continue
        
        init = init_map[onnx_name]
        weight = numpy_helper.to_array(init)
        
        down = lora_tensors[base + ".down"]
        up = lora_tensors[base + ".up"]
        alpha = lora_alphas.get(base, 1.0)
        
        # Compute delta W = alpha * (up @ down)
        # Handle different shapes
        if down.ndim == 4 and up.ndim == 4:
            # Conv2d: down is (rank, in_ch, 1, 1), up is (out_ch, rank, 1, 1)
            delta = (up.squeeze() @ down.squeeze()).reshape(up.shape[0], down.shape[1], 1, 1)
        elif down.ndim == 2 and up.ndim == 2:
            # Linear: down is (rank, in_features), up is (out_features, rank)
            delta = up @ down
        else:
            print(f"  Unexpected shapes for {base}: down={down.shape}, up={up.shape}")
            skipped += 1
            continue
        
        delta = alpha * delta
        
        # Broadcast delta to match weight shape if needed
        if delta.shape != weight.shape:
            # Try broadcasting
            try:
                delta = np.broadcast_to(delta, weight.shape).copy()
            except ValueError:
                print(f"  Shape mismatch for {onnx_name}: weight={weight.shape}, delta={delta.shape}")
                skipped += 1
                continue
        
        # Apply: W = W + delta
        new_weight = weight + delta
        
        # Update initializer
        new_init = numpy_helper.from_array(new_weight, name=onnx_name)
        # Replace in model
        for i, init in enumerate(model.graph.initializer):
            if init.name == onnx_name:
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
