#!/usr/bin/env python3
"""
MT3 Model Validation Script

Script to validate the MT3 model implementation, test forward passes,
and verify compatibility with expected parameter shapes.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any

from mt3_model import MT3Model, MT3Config, create_mt3_model
from checkpoint_utils import (
    create_parameter_mapping_report,
    diagnose_checkpoint_compatibility,
    validate_parameter_names
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test basic model creation and initialization."""
    logger.info("Testing model creation...")

    # Test with default config
    config = MT3Config()
    model = MT3Model(config)

    # Verify configuration
    assert model.config.vocab_size == 1536
    assert model.config.d_model == 512
    assert model.config.num_encoder_layers == 8
    assert model.config.num_decoder_layers == 8
    assert model.config.num_heads == 8
    assert model.config.d_ff == 1024

    # Check parameter count (approximately 45.8M)
    param_count = model.get_parameter_count()
    expected_count = 45_800_000  # Approximately 45.8M
    tolerance = 1_000_000  # 1M parameter tolerance

    assert abs(param_count - expected_count) < tolerance, f"Parameter count {param_count} not close to expected {expected_count}"

    logger.info(f"✅ Model creation successful. Parameters: {param_count:,}")
    return model


def test_forward_pass(model: MT3Model):
    """Test forward pass with dummy data."""
    logger.info("Testing forward pass...")

    batch_size = 2
    encoder_seq_len = 256
    decoder_seq_len = 32
    vocab_size = model.config.vocab_size

    # Create dummy inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))

    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

    # Verify output shapes
    expected_logits_shape = (batch_size, decoder_seq_len, vocab_size)
    assert outputs['logits'].shape == expected_logits_shape, f"Logits shape {outputs['logits'].shape} != expected {expected_logits_shape}"

    # Verify other outputs
    assert outputs['encoder_last_hidden_state'].shape == (batch_size, encoder_seq_len, model.config.d_model)

    logger.info(f"✅ Forward pass successful. Logits shape: {outputs['logits'].shape}")
    return outputs


def test_generation(model: MT3Model):
    """Test autoregressive generation."""
    logger.info("Testing generation...")

    batch_size = 1
    encoder_seq_len = 128
    max_length = 50

    # Create dummy encoder input
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, encoder_seq_len))

    # Test generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=False,  # Greedy decoding
            early_stopping=True
        )

    # Verify output shape
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= max_length
    assert generated.shape[1] > 1  # Should generate at least one token beyond start token

    logger.info(f"✅ Generation successful. Generated shape: {generated.shape}")
    return generated


def test_parameter_shapes(model: MT3Model):
    """Test that parameter shapes match expected T5 format."""
    logger.info("Testing parameter shapes...")

    state_dict = model.state_dict()
    config = model.config

    # Expected shapes for key parameters
    expected_shapes = {
        'shared.weight': [config.vocab_size, config.d_model],
        'lm_head.weight': [config.vocab_size, config.d_model],
        'encoder.block.0.layer.0.SelfAttention.q.weight': [config.num_heads * config.d_kv, config.d_model],
        'encoder.block.0.layer.0.SelfAttention.k.weight': [config.num_heads * config.d_kv, config.d_model],
        'encoder.block.0.layer.0.SelfAttention.v.weight': [config.num_heads * config.d_kv, config.d_model],
        'encoder.block.0.layer.0.SelfAttention.o.weight': [config.d_model, config.num_heads * config.d_kv],
        'encoder.block.0.layer.1.DenseReluDense.wi_0.weight': [config.d_ff, config.d_model],
        'encoder.block.0.layer.1.DenseReluDense.wi_1.weight': [config.d_ff, config.d_model],
        'encoder.block.0.layer.1.DenseReluDense.wo.weight': [config.d_model, config.d_ff],
    }

    mismatches = []
    for param_name, expected_shape in expected_shapes.items():
        if param_name in state_dict:
            actual_shape = list(state_dict[param_name].shape)
            if actual_shape != expected_shape:
                mismatches.append(f"{param_name}: expected {expected_shape}, got {actual_shape}")
        else:
            mismatches.append(f"{param_name}: parameter not found")

    if mismatches:
        logger.warning("Parameter shape mismatches found:")
        for mismatch in mismatches:
            logger.warning(f"  {mismatch}")
    else:
        logger.info("✅ All parameter shapes match expected values")

    return len(mismatches) == 0


def test_checkpoint_compatibility(checkpoint_path: str = None):
    """Test checkpoint loading if checkpoint is available."""
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        logger.info("No checkpoint provided or found, skipping checkpoint tests")
        return True

    logger.info(f"Testing checkpoint compatibility: {checkpoint_path}")

    try:
        # Analyze checkpoint
        analysis = diagnose_checkpoint_compatibility(checkpoint_path)

        logger.info(f"Checkpoint analysis:")
        logger.info(f"  File size: {analysis['file_size_mb']:.1f} MB")
        logger.info(f"  Total parameters: {analysis['total_parameters']:,}")
        logger.info(f"  Parameter count: {analysis['parameter_count']}")

        if analysis['inferred_config']['vocab_size']:
            logger.info(f"  Inferred vocab_size: {analysis['inferred_config']['vocab_size']}")
        if analysis['inferred_config']['d_model']:
            logger.info(f"  Inferred d_model: {analysis['inferred_config']['d_model']}")

        # Test loading
        model = MT3Model(MT3Config())
        missing_keys, unexpected_keys = validate_parameter_names(model, checkpoint_path)

        logger.info(f"  Missing keys: {len(missing_keys)}")
        logger.info(f"  Unexpected keys: {len(unexpected_keys)}")

        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            logger.info("✅ Perfect checkpoint compatibility")
            return True
        else:
            logger.warning("⚠️ Checkpoint has some compatibility issues")
            return False

    except Exception as e:
        logger.error(f"❌ Checkpoint test failed: {e}")
        return False


def run_comprehensive_validation(checkpoint_path: str = None) -> Dict[str, Any]:
    """Run comprehensive validation of the MT3 model."""
    logger.info("🚀 Starting comprehensive MT3 model validation")

    results = {
        'model_creation': False,
        'forward_pass': False,
        'generation': False,
        'parameter_shapes': False,
        'checkpoint_compatibility': None,
        'errors': []
    }

    try:
        # Test 1: Model creation
        model = test_model_creation()
        results['model_creation'] = True

        # Test 2: Forward pass
        outputs = test_forward_pass(model)
        results['forward_pass'] = True

        # Test 3: Generation
        generated = test_generation(model)
        results['generation'] = True

        # Test 4: Parameter shapes
        shapes_ok = test_parameter_shapes(model)
        results['parameter_shapes'] = shapes_ok

        # Test 5: Checkpoint compatibility (if available)
        if checkpoint_path:
            checkpoint_ok = test_checkpoint_compatibility(checkpoint_path)
            results['checkpoint_compatibility'] = checkpoint_ok

    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        results['errors'].append(str(e))

    # Summary
    logger.info("📊 Validation Summary:")
    logger.info(f"  ✅ Model creation: {'PASS' if results['model_creation'] else 'FAIL'}")
    logger.info(f"  ✅ Forward pass: {'PASS' if results['forward_pass'] else 'FAIL'}")
    logger.info(f"  ✅ Generation: {'PASS' if results['generation'] else 'FAIL'}")
    logger.info(f"  ✅ Parameter shapes: {'PASS' if results['parameter_shapes'] else 'FAIL'}")

    if results['checkpoint_compatibility'] is not None:
        logger.info(f"  ✅ Checkpoint compatibility: {'PASS' if results['checkpoint_compatibility'] else 'FAIL'}")

    if results['errors']:
        logger.error(f"  ❌ Errors: {len(results['errors'])}")
        for error in results['errors']:
            logger.error(f"    - {error}")

    return results


def create_validation_report(results: Dict[str, Any], output_path: str):
    """Create a detailed validation report."""
    lines = [
        "MT3 Model Validation Report",
        "=" * 50,
        "",
        f"Model Creation: {'PASS' if results['model_creation'] else 'FAIL'}",
        f"Forward Pass: {'PASS' if results['forward_pass'] else 'FAIL'}",
        f"Generation: {'PASS' if results['generation'] else 'FAIL'}",
        f"Parameter Shapes: {'PASS' if results['parameter_shapes'] else 'FAIL'}",
        ""
    ]

    if results['checkpoint_compatibility'] is not None:
        lines.append(f"Checkpoint Compatibility: {'PASS' if results['checkpoint_compatibility'] else 'FAIL'}")
        lines.append("")

    if results['errors']:
        lines.extend([
            "Errors:",
            "-" * 20
        ])
        for error in results['errors']:
            lines.append(f"- {error}")
        lines.append("")

    # Model specifications
    lines.extend([
        "Model Specifications:",
        "-" * 20,
        "- Architecture: T5 encoder-decoder",
        "- Encoder layers: 8",
        "- Decoder layers: 8",
        "- d_model: 512",
        "- num_heads: 8",
        "- d_ff: 1024",
        "- vocab_size: 1536",
        "- Parameter count: ~45.8M",
        ""
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate MT3 model implementation")
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint file for testing")
    parser.add_argument('--output-report', type=str, default="mt3_validation_report.txt",
                       help="Path to save validation report")
    parser.add_argument('--create-param-report', action='store_true',
                       help="Create parameter mapping report")

    args = parser.parse_args()

    # Run validation
    results = run_comprehensive_validation(args.checkpoint)

    # Create reports
    create_validation_report(results, args.output_report)

    if args.create_param_report:
        model = create_mt3_model()
        create_parameter_mapping_report(model, "mt3_parameter_mapping.txt")

    # Exit with appropriate code
    all_tests_passed = (
        results['model_creation'] and
        results['forward_pass'] and
        results['generation'] and
        results['parameter_shapes'] and
        (results['checkpoint_compatibility'] is None or results['checkpoint_compatibility'])
    )

    if all_tests_passed:
        logger.info("🎉 All validation tests passed!")
        exit(0)
    else:
        logger.error("💥 Some validation tests failed!")
        exit(1)