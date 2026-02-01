#!/usr/bin/env python3
"""
Hamburger Model v2 - With Stitching Layers

Based on Neural Organ Transplantation (NOT) paper:
- Adds learnable stitching layers between host and donor
- Stitching layers bridge representational gaps
- Additive insertion preserves base model capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from hamburger_transplant.utils import get_device, ensure_organ_exists


@dataclass
class InsertedOrgan:
    """Configuration for an inserted organ layer."""
    name: str
    insert_after_layer: int
    organ_path: str
    hidden_size: int = 3584
    scale: float = 0.1


class StitchingLayer(nn.Module):
    """
    Learnable linear transformation to bridge representational gaps.
    Based on Model Stitching (Bansal et al., NeurIPS 2021).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=True)
        if in_features == out_features:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            nn.init.normal_(self.proj.weight, std=0.02)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OrganLayerV2(nn.Module):
    """
    Organ layer with stitching layers for proper integration.

    Architecture:
        host_hidden -> stitch_in -> LayerNorm -> MLP -> stitch_out -> + residual
    """

    def __init__(self, organ_path: str, hidden_size: int = 3584, scale: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.hidden_size = hidden_size

        # Load organ weights
        organ_file = Path(organ_path)
        ensure_organ_exists(organ_file, organ_file.stem)

        data = torch.load(organ_path, map_location='cpu', weights_only=False)
        self.organ_hidden = data['config']['hidden_size']
        self.organ_inter = data['config']['intermediate_size']

        # Stitching layers (LEARNABLE)
        self.stitch_in = StitchingLayer(hidden_size, self.organ_hidden)
        self.stitch_out = StitchingLayer(self.organ_hidden, hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(self.organ_hidden)

        # MLP (from organ, FROZEN)
        self.gate_proj = nn.Linear(self.organ_hidden, self.organ_inter, bias=False)
        self.up_proj = nn.Linear(self.organ_hidden, self.organ_inter, bias=False)
        self.down_proj = nn.Linear(self.organ_inter, self.organ_hidden, bias=False)

        # Load weights
        weights = data['weights']
        for key, value in weights.items():
            if 'layers.0.mlp.gate_proj' in key:
                self.gate_proj.weight.data = value.half()
            elif 'layers.0.mlp.up_proj' in key:
                self.up_proj.weight.data = value.half()
            elif 'layers.0.mlp.down_proj' in key:
                self.down_proj.weight.data = value.half()
            elif 'layers.0.input_layernorm' in key or 'layers.0.post_attention_layernorm' in key:
                if hasattr(self.norm, 'weight') and self.norm.weight.shape == value.shape:
                    self.norm.weight.data = value.half()

        # Freeze organ MLP weights
        self.gate_proj.weight.requires_grad = False
        self.up_proj.weight.requires_grad = False
        self.down_proj.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Stitch IN: host representation -> organ representation
        h = self.stitch_in(x)

        # Layer norm
        h = self.norm(h)

        # SwiGLU MLP (frozen organ)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)

        # Stitch OUT: organ representation -> host representation
        out = self.stitch_out(mlp_out)

        # Residual connection with learnable scale
        return residual + self.scale * out

    def get_trainable_params(self):
        """Return only trainable parameters (stitching layers + scale)."""
        params = []
        params.extend(self.stitch_in.parameters())
        params.extend(self.stitch_out.parameters())
        params.append(self.scale)
        return params


class HamburgerModelV2(nn.Module):
    """
    Hamburger Model v2 with trainable stitching layers.

    Inserts frozen organ MLP layers between host transformer layers,
    connected via learnable stitching layers and residual connections.
    """

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        organ_insertions: Optional[List[InsertedOrgan]] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or get_device()

        print("=" * 70)
        print("Hamburger Model v2 - With Stitching Layers")
        print("=" * 70)

        # Load base model
        print(f"\n[1] Loading base model: {base_model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        self.hidden_size = self.base_model.config.hidden_size
        self.num_base_layers = self.base_model.config.num_hidden_layers

        print(f"  Base layers: {self.num_base_layers}")
        print(f"  Hidden size: {self.hidden_size}")

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Default organ insertions
        if organ_insertions is None:
            organ_insertions = []

        # Create organ layers with stitching
        print(f"\n[2] Creating organ layers with stitching...")
        self.organ_layers = nn.ModuleDict()
        self.insertion_points = {}

        for organ_config in organ_insertions:
            print(f"  Adding {organ_config.name} after layer {organ_config.insert_after_layer}")

            organ_layer = OrganLayerV2(
                organ_path=organ_config.organ_path,
                hidden_size=self.hidden_size,
                scale=organ_config.scale,
            )

            self.organ_layers[organ_config.name] = organ_layer.to(self.device).float()
            self.insertion_points[organ_config.insert_after_layer] = organ_config.name

        # Setup hooks
        self._setup_hooks()

        # Print architecture
        self._print_architecture()

        # Count trainable params
        total, trainable = self._count_params()
        print(f"\n  Trainable parameters: {trainable:,} / {total:,}")

        print("\n" + "=" * 70)
        print("Hamburger Model v2 Ready!")
        print("=" * 70)

    def _setup_hooks(self):
        """Setup forward hooks to insert organ layers."""
        for layer_idx, organ_name in self.insertion_points.items():
            organ_layer = self.organ_layers[organ_name]

            def make_hook(organ, name, idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    original_dtype = hidden.dtype

                    # Process in float32 for training stability
                    hidden_f32 = hidden.float()
                    organ_output = organ(hidden_f32)

                    # Clamp to prevent nan/inf (preserves gradient path)
                    if torch.isnan(organ_output).any() or torch.isinf(organ_output).any():
                        print(f"[{name}] WARNING: nan/inf detected, clamping")
                        organ_output = torch.nan_to_num(organ_output, nan=0.0, posinf=1e4, neginf=-1e4)

                    organ_output = organ_output.to(original_dtype)

                    if isinstance(output, tuple):
                        return (organ_output,) + output[1:]
                    return organ_output

                return hook

            hook_fn = make_hook(organ_layer, organ_name, layer_idx)
            self.base_model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def _print_architecture(self):
        """Print the hamburger architecture."""
        print(f"\n[3] Architecture:")
        print("  " + "-" * 50)

        for i in range(self.num_base_layers):
            print(f"  | Layer {i:2d}: Base Model")
            if i in self.insertion_points:
                organ_name = self.insertion_points[i]
                print(f"  | -------- {organ_name.upper()} ORGAN (stitched) --------")

        print("  " + "-" * 50)
        print(f"  Total effective layers: {self.num_base_layers} + {len(self.insertion_points)} organs")

    def _count_params(self):
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_trainable_params(self):
        """Get all trainable parameters (stitching layers only)."""
        params = []
        for organ in self.organ_layers.values():
            params.extend(organ.get_trainable_params())
        return params

    def save_stitching(self, path: str):
        """Save only the stitching layer weights."""
        state = {}
        for name, organ in self.organ_layers.items():
            state[name] = {
                'stitch_in': organ.stitch_in.state_dict(),
                'stitch_out': organ.stitch_out.state_dict(),
                'scale': organ.scale.data,
            }
        torch.save(state, path)
        print(f"Saved stitching weights to {path}")

    def load_stitching(self, path: str):
        """Load stitching layer weights."""
        state = torch.load(path, map_location='cpu', weights_only=False)
        for name, organ in self.organ_layers.items():
            if name in state:
                organ.stitch_in.load_state_dict(state[name]['stitch_in'])
                organ.stitch_out.load_state_dict(state[name]['stitch_out'])
                organ.scale.data = state[name]['scale']
        print(f"Loaded stitching weights from {path}")

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.base_model.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = inputs["input_ids"].shape[-1]
        response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return response
