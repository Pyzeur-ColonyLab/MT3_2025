#!/usr/bin/env python3
"""
Checkpoint Utilities for MT3 Model

Utilities for loading and validating converted MT3 checkpoints,
including parameter name mapping and state dict compatibility.
"""

import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .mt3_model import MT3Model, MT3Config

logger = logging.getLogger(__name__)


def validate_parameter_names(model: MT3Model, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    """
    Validate parameter names compatibility between model and checkpoint.

    Args:
        model: MT3Model instance
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Get model parameter names
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Find differences
    missing_keys = list(model_keys - checkpoint_keys)
    unexpected_keys = list(checkpoint_keys - model_keys)

    return missing_keys, unexpected_keys


def load_mt3_checkpoint(
    model: MT3Model,
    checkpoint_path: str,
    strict: bool = True,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load MT3 checkpoint with proper error handling and validation.

    Args:
        model: MT3Model instance to load checkpoint into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
        map_location: Device to map tensors to

    Returns:
        Dictionary with loading results and metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    device = map_location or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    else:
        state_dict = checkpoint
        metadata = {}

    # Log checkpoint info
    logger.info(f"Checkpoint contains {len(state_dict)} parameter tensors")
    if metadata:
        logger.info(f"Metadata keys: {list(metadata.keys())}")

    # Validate parameter compatibility
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing_keys = list(model_keys - checkpoint_keys)
    unexpected_keys = list(checkpoint_keys - model_keys)

    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    # Load state dict
    try:
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        logger.info("Checkpoint loaded successfully")

        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            logger.warning(f"Incompatible keys - Missing: {len(incompatible_keys.missing_keys)}, "
                         f"Unexpected: {len(incompatible_keys.unexpected_keys)}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    return {
        'success': True,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'metadata': metadata,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'checkpoint_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
    }


def create_parameter_mapping_report(model: MT3Model, output_path: str) -> None:
    """
    Create a detailed parameter mapping report for debugging.

    Args:
        model: MT3Model instance
        output_path: Path to save the report
    """
    report_lines = ["MT3 Model Parameter Mapping Report", "=" * 50, ""]

    # Model architecture summary
    config = model.config
    report_lines.extend([
        "Model Configuration:",
        f"  vocab_size: {config.vocab_size}",
        f"  d_model: {config.d_model}",
        f"  num_encoder_layers: {config.num_encoder_layers}",
        f"  num_decoder_layers: {config.num_decoder_layers}",
        f"  num_heads: {config.num_heads}",
        f"  d_ff: {config.d_ff}",
        f"  d_kv: {config.d_kv}",
        ""
    ])

    # Parameter details
    total_params = 0
    report_lines.append("Parameter Details:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Parameter Name':<60} {'Shape':<20} {'Count':>10}")
    report_lines.append("-" * 80)

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        shape_str = str(list(param.shape))
        report_lines.append(f"{name:<60} {shape_str:<20} {param_count:>10,}")

    report_lines.extend([
        "-" * 80,
        f"{'Total Parameters':<60} {'':<20} {total_params:>10,}",
        ""
    ])

    # Expected shapes for key parameters
    report_lines.extend([
        "Expected Parameter Shapes (for checkpoint validation):",
        "-" * 50,
        "Encoder:",
        "  encoder.block.0.layer.0.SelfAttention.q.weight: [384, 512]",
        "  encoder.block.0.layer.0.SelfAttention.k.weight: [384, 512]",
        "  encoder.block.0.layer.0.SelfAttention.v.weight: [384, 512]",
        "  encoder.block.0.layer.1.DenseReluDense.wi_0.weight: [1024, 512]",
        "",
        "Decoder:",
        "  decoder.block.0.layer.0.SelfAttention.q.weight: [384, 512]",
        "  decoder.block.0.layer.1.EncDecAttention.k.weight: [384, 512]",
        "  decoder.block.0.layer.2.DenseReluDense.wi_0.weight: [1024, 512]",
        "",
        "Embeddings and Output:",
        f"  shared.weight: [{config.vocab_size}, {config.d_model}]",
        f"  lm_head.weight: [{config.vocab_size}, {config.d_model}]",
        ""
    ])

    # Component breakdown
    param_summary = model.get_parameter_summary()
    report_lines.extend([
        "Component Parameter Breakdown:",
        "-" * 30,
        f"Shared Embeddings: {param_summary['shared_embeddings']:,} ({param_summary['shared_embeddings']/total_params*100:.1f}%)",
        f"Encoder: {param_summary['encoder']:,} ({param_summary['encoder']/total_params*100:.1f}%)",
        f"Decoder: {param_summary['decoder']:,} ({param_summary['decoder']/total_params*100:.1f}%)",
        f"LM Head: {param_summary['lm_head']:,} ({param_summary['lm_head']/total_params*100:.1f}%)",
        f"Total: {total_params:,}",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Parameter mapping report saved to {output_path}")


def diagnose_checkpoint_compatibility(
    checkpoint_path: str,
    config: Optional[MT3Config] = None
) -> Dict[str, Any]:
    """
    Diagnose checkpoint compatibility without loading into model.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional model configuration

    Returns:
        Dictionary with compatibility analysis
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Analyze checkpoint structure
    analysis = {
        'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
        'total_parameters': sum(p.numel() for p in state_dict.values()),
        'parameter_count': len(state_dict),
        'parameter_names': list(state_dict.keys()),
        'parameter_shapes': {k: list(v.shape) for k, v in state_dict.items()},
    }

    # Check for expected parameter patterns
    expected_patterns = [
        'shared.weight',
        'encoder.block.0.layer.0.SelfAttention.q.weight',
        'decoder.block.0.layer.1.EncDecAttention.k.weight',
        'lm_head.weight',
    ]

    pattern_matches = {}
    for pattern in expected_patterns:
        matches = [k for k in state_dict.keys() if pattern in k]
        pattern_matches[pattern] = matches

    analysis['pattern_matches'] = pattern_matches

    # Infer configuration from checkpoint
    if 'shared.weight' in state_dict:
        inferred_vocab_size, inferred_d_model = state_dict['shared.weight'].shape
    else:
        # Try to infer from other parameters
        inferred_vocab_size, inferred_d_model = None, None

    analysis['inferred_config'] = {
        'vocab_size': inferred_vocab_size,
        'd_model': inferred_d_model,
    }

    # Check compatibility with provided config
    if config is not None:
        compatibility_issues = []

        if inferred_vocab_size and inferred_vocab_size != config.vocab_size:
            compatibility_issues.append(
                f"Vocab size mismatch: checkpoint={inferred_vocab_size}, config={config.vocab_size}"
            )

        if inferred_d_model and inferred_d_model != config.d_model:
            compatibility_issues.append(
                f"Model dimension mismatch: checkpoint={inferred_d_model}, config={config.d_model}"
            )

        analysis['compatibility_issues'] = compatibility_issues
        analysis['is_compatible'] = len(compatibility_issues) == 0
    else:
        analysis['compatibility_issues'] = []
        analysis['is_compatible'] = True

    return analysis


def convert_t5x_parameter_names(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert T5X parameter names to PyTorch MT3 format.

    This function handles the parameter name mapping from T5X checkpoints
    to the expected PyTorch parameter naming convention.

    Args:
        state_dict: Original state dict with T5X parameter names

    Returns:
        Converted state dict with PyTorch parameter names
    """
    # This is a placeholder for T5X -> PyTorch name mapping
    # The actual mapping would depend on the specific T5X checkpoint format

    name_mapping = {
        # Example mappings (these would need to be determined from actual checkpoint)
        'encoder/embeddings/token': 'shared.weight',
        'decoder/embeddings/token': 'shared.weight',  # Shared embeddings
        'decoder/logits_dense': 'lm_head.weight',

        # Attention layer mappings
        'encoder/layers_0/attention/query': 'encoder.block.0.layer.0.SelfAttention.q.weight',
        'encoder/layers_0/attention/key': 'encoder.block.0.layer.0.SelfAttention.k.weight',
        'encoder/layers_0/attention/value': 'encoder.block.0.layer.0.SelfAttention.v.weight',
        'encoder/layers_0/attention/out': 'encoder.block.0.layer.0.SelfAttention.o.weight',

        # Feed-forward mappings
        'encoder/layers_0/mlp/wi_0': 'encoder.block.0.layer.1.DenseReluDense.wi_0.weight',
        'encoder/layers_0/mlp/wi_1': 'encoder.block.0.layer.1.DenseReluDense.wi_1.weight',
        'encoder/layers_0/mlp/wo': 'encoder.block.0.layer.1.DenseReluDense.wo.weight',
    }

    converted_state_dict = {}

    for original_name, tensor in state_dict.items():
        # Apply name mapping if available
        if original_name in name_mapping:
            new_name = name_mapping[original_name]
            converted_state_dict[new_name] = tensor
            logger.info(f"Mapped {original_name} -> {new_name}")
        else:
            # Keep original name if no mapping found
            converted_state_dict[original_name] = tensor
            logger.warning(f"No mapping found for parameter: {original_name}")

    return converted_state_dict


# Example usage functions
def create_model_from_checkpoint(checkpoint_path: str, config: Optional[MT3Config] = None) -> MT3Model:
    """
    Create MT3Model and load checkpoint in one step.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional model configuration

    Returns:
        MT3Model with loaded checkpoint
    """
    if config is None:
        # Try to infer config from checkpoint
        analysis = diagnose_checkpoint_compatibility(checkpoint_path)
        inferred_config = analysis['inferred_config']

        if inferred_config['vocab_size'] and inferred_config['d_model']:
            config = MT3Config(
                vocab_size=inferred_config['vocab_size'],
                d_model=inferred_config['d_model']
            )
            logger.info(f"Inferred config from checkpoint: vocab_size={config.vocab_size}, d_model={config.d_model}")
        else:
            config = MT3Config()  # Use default
            logger.warning("Could not infer config from checkpoint, using default")

    # Create model
    model = MT3Model(config)

    # Load checkpoint
    result = load_mt3_checkpoint(model, checkpoint_path, strict=False)

    if result['success']:
        logger.info("Model created and checkpoint loaded successfully")
    else:
        logger.warning("Model created but checkpoint loading had issues")

    return model


if __name__ == "__main__":
    # Example usage
    config = MT3Config()
    model = MT3Model(config)

    # Create parameter mapping report
    create_parameter_mapping_report(model, "mt3_parameter_mapping.txt")
    print("Parameter mapping report created: mt3_parameter_mapping.txt")