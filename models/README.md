# MT3 PyTorch Implementation

Production-ready PyTorch implementation of MT3 (Music Transcription with Transformers) based on the T5 architecture, designed for compatibility with converted T5X checkpoints.

## Overview

This implementation provides a complete MT3 model that matches the original specifications:

- **Architecture**: T5 encoder-decoder with 8 layers each
- **Parameters**: ~45.8M parameters total
- **Specifications**:
  - `d_model`: 512
  - `vocab_size`: 1536
  - `num_heads`: 8
  - `d_ff`: 1024
  - `d_kv`: 64

## Files Structure

```
MT3/models/
├── __init__.py                 # Package initialization
├── mt3_model.py               # Core MT3Model implementation
├── checkpoint_utils.py        # Checkpoint loading utilities
├── validate_model.py          # Model validation script
├── test_syntax.py            # Syntax validation (no PyTorch needed)
└── README.md                 # This file
```

## Key Features

### Architecture Components

- **T5-style relative position bias** for attention mechanisms
- **Shared embedding layer** between encoder and decoder
- **Cross-attention layers** in decoder blocks
- **RMSNorm** layer normalization (T5 standard)
- **Gated activation** in feed-forward layers
- **Autoregressive generation** with multiple strategies

### Parameter Compatibility

The implementation uses parameter names compatible with converted T5X checkpoints:

```python
# Example parameter mappings:
encoder.block.0.layer.0.SelfAttention.q.weight  # Shape: [384, 512]
decoder.block.0.layer.1.EncDecAttention.k.weight # Shape: [384, 512]
decoder.block.0.layer.2.DenseReluDense.wi_0.weight # Shape: [1024, 512]
shared.weight                                     # Shape: [1536, 512]
lm_head.weight                                   # Shape: [1536, 512]
```

## Quick Start

### 1. Basic Usage

```python
from MT3.models import MT3Model, MT3Config, create_mt3_model

# Create model with default configuration
model = create_mt3_model()

# Or create with custom config
config = MT3Config(
    vocab_size=1536,
    d_model=512,
    num_encoder_layers=8,
    num_decoder_layers=8,
    num_heads=8,
    d_ff=1024
)
model = MT3Model(config)
```

### 2. Forward Pass

```python
import torch

# Prepare inputs
batch_size = 2
encoder_seq_len = 256
decoder_seq_len = 32

input_ids = torch.randint(0, 1536, (batch_size, encoder_seq_len))
decoder_input_ids = torch.randint(0, 1536, (batch_size, decoder_seq_len))

# Forward pass
outputs = model(
    input_ids=input_ids,
    decoder_input_ids=decoder_input_ids,
    return_dict=True
)

print(f"Logits shape: {outputs['logits'].shape}")  # [2, 32, 1536]
```

### 3. Generation

```python
# Autoregressive generation
generated = model.generate(
    input_ids=input_ids[:1],  # Single example
    max_length=50,
    do_sample=False,          # Greedy decoding
    temperature=1.0
)

print(f"Generated tokens: {generated.shape}")  # [1, <=50]
```

## Checkpoint Loading

### Basic Loading

```python
from MT3.models.checkpoint_utils import load_mt3_checkpoint

# Load checkpoint into existing model
result = load_mt3_checkpoint(
    model=model,
    checkpoint_path="path/to/mt3_converted.pth",
    strict=False  # Allow missing/unexpected keys
)

print(f"Loading successful: {result['success']}")
print(f"Missing keys: {len(result['missing_keys'])}")
print(f"Unexpected keys: {len(result['unexpected_keys'])}")
```

### One-Step Creation and Loading

```python
from MT3.models.checkpoint_utils import create_model_from_checkpoint

# Create model and load checkpoint in one step
model = create_model_from_checkpoint("path/to/mt3_converted.pth")
```

### Checkpoint Compatibility Analysis

```python
from MT3.models.checkpoint_utils import diagnose_checkpoint_compatibility

# Analyze checkpoint without loading
analysis = diagnose_checkpoint_compatibility("path/to/mt3_converted.pth")

print(f"File size: {analysis['file_size_mb']:.1f} MB")
print(f"Parameter count: {analysis['total_parameters']:,}")
print(f"Inferred vocab_size: {analysis['inferred_config']['vocab_size']}")
print(f"Compatible: {analysis['is_compatible']}")
```

## Validation and Testing

### Syntax Validation (No PyTorch Required)

```bash
cd MT3/models
python test_syntax.py
```

### Full Model Validation

```bash
# Install PyTorch first
pip install torch

# Run comprehensive validation
python validate_model.py --create-param-report

# Test with checkpoint
python validate_model.py --checkpoint path/to/checkpoint.pth
```

### Parameter Mapping Report

```python
from MT3.models.checkpoint_utils import create_parameter_mapping_report

create_parameter_mapping_report(model, "parameter_report.txt")
```

## Model Architecture Details

### Encoder Stack
- 8 transformer blocks
- Each block contains:
  - Multi-head self-attention with relative position bias
  - Feed-forward layer with gated activation
  - RMSNorm layer normalization
  - Residual connections

### Decoder Stack
- 8 transformer blocks
- Each block contains:
  - Masked self-attention with relative position bias
  - Cross-attention to encoder outputs
  - Feed-forward layer with gated activation
  - RMSNorm layer normalization
  - Residual connections

### Attention Mechanism
- **Heads**: 8
- **Head dimension**: 64 (d_model / num_heads)
- **Relative position bias**: T5-style bucketed relative positions
- **Attention dropout**: 0.1 (configurable)

### Feed-Forward Networks
- **Hidden size**: 1024 (d_ff)
- **Activation**: Gated ReLU (wi_0 * ReLU(wi_1))
- **Dropout**: 0.1 (configurable)

## Configuration Options

```python
@dataclass
class MT3Config:
    # Model architecture
    vocab_size: int = 1536
    d_model: int = 512
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    num_heads: int = 8
    d_ff: int = 1024
    d_kv: int = 64

    # Training parameters
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6

    # Position encoding
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    # Special tokens
    decoder_start_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 0

    # Generation
    max_length: int = 1024
```

## Generation Options

The model supports various generation strategies:

```python
# Greedy decoding
tokens = model.generate(
    input_ids=input_ids,
    max_length=100,
    do_sample=False
)

# Temperature sampling
tokens = model.generate(
    input_ids=input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.8
)

# Top-k sampling
tokens = model.generate(
    input_ids=input_ids,
    max_length=100,
    do_sample=True,
    top_k=50
)

# Top-p (nucleus) sampling
tokens = model.generate(
    input_ids=input_ids,
    max_length=100,
    do_sample=True,
    top_p=0.9
)
```

## Performance and Memory

### Parameter Count
- **Total**: ~45.8M parameters
- **Shared Embeddings**: ~786K parameters (1536 × 512)
- **Encoder**: ~22.5M parameters
- **Decoder**: ~22.5M parameters
- **LM Head**: Shares weights with embeddings

### Memory Requirements
- **Model**: ~180 MB (float32)
- **Training**: ~1-2 GB (depending on batch size)
- **Inference**: ~500 MB (including activations)

### Optimization Tips

1. **Use torch.cuda.amp** for mixed precision training
2. **Gradient checkpointing** for memory-efficient training
3. **Model.eval()** during inference to disable dropout
4. **torch.no_grad()** context for inference
5. **Batch processing** for efficient inference

## Error Handling

The implementation includes comprehensive error checking:

```python
# Configuration validation
try:
    config = MT3Config(d_model=511, num_heads=8)  # Invalid: not divisible
except ValueError as e:
    print(f"Configuration error: {e}")

# Checkpoint loading validation
result = load_mt3_checkpoint(model, "checkpoint.pth", strict=False)
if not result['success']:
    print("Checkpoint loading failed!")
    print(f"Missing keys: {result['missing_keys']}")
    print(f"Unexpected keys: {result['unexpected_keys']}")

# Generation validation
try:
    tokens = model.generate(input_ids=None)  # Missing required input
except ValueError as e:
    print(f"Generation error: {e}")
```

## Integration with MT3 Pipeline

This model is designed to work with the broader MT3 ecosystem:

1. **Audio preprocessing**: Convert audio to spectrograms
2. **Model inference**: Use this PyTorch implementation
3. **Token decoding**: Convert tokens back to MIDI events
4. **MIDI generation**: Create final MIDI files

```python
# Typical usage in MT3 pipeline
from MT3.models import create_mt3_model
from MT3.models.checkpoint_utils import load_mt3_checkpoint

# 1. Load model
model = create_mt3_model()
load_mt3_checkpoint(model, "mt3_converted.pth")

# 2. Process audio (external preprocessing)
audio_features = preprocess_audio(audio_file)  # Your preprocessing

# 3. Generate tokens
tokens = model.generate(
    input_ids=audio_features,
    max_length=1024,
    do_sample=False
)

# 4. Decode to MIDI (external postprocessing)
midi_file = decode_tokens_to_midi(tokens, vocab)  # Your postprocessing
```

## Requirements

### Core Dependencies
```
torch>=1.9.0
numpy>=1.20.0
```

### Optional Dependencies for Full Pipeline
```
librosa          # Audio processing
pretty_midi      # MIDI handling
note_seq         # Music sequence utilities
```

### Development Dependencies
```
pytest           # Testing
pytest-benchmark # Performance testing
```

## Troubleshooting

### Common Issues

1. **Parameter count mismatch**
   - Check configuration matches checkpoint
   - Verify shared embeddings are properly tied
   - Run parameter mapping report for debugging

2. **Checkpoint loading fails**
   - Use `strict=False` to allow missing keys
   - Check parameter name compatibility
   - Run checkpoint compatibility analysis

3. **Out of memory during training**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

4. **Generation produces invalid tokens**
   - Check vocab_size matches model configuration
   - Verify EOS/PAD token IDs are correct
   - Validate input preprocessing

### Debug Tools

```python
# Parameter summary
summary = model.get_parameter_summary()
print(f"Total parameters: {summary['total']:,}")

# Parameter shapes report
from MT3.models.checkpoint_utils import create_parameter_mapping_report
create_parameter_mapping_report(model, "debug_params.txt")

# Checkpoint analysis
from MT3.models.checkpoint_utils import diagnose_checkpoint_compatibility
analysis = diagnose_checkpoint_compatibility("checkpoint.pth")
print(f"Compatibility issues: {analysis['compatibility_issues']}")
```

## Contributing

When contributing to this implementation:

1. **Maintain compatibility** with T5 architecture standards
2. **Preserve parameter names** for checkpoint compatibility
3. **Add comprehensive tests** for new features
4. **Update documentation** for any API changes
5. **Validate against original MT3** when possible

## License

This implementation follows the same license as the original MT3 project.

## Acknowledgments

- Based on the original MT3 paper and implementation by Magenta team
- T5 architecture implementation inspired by Hugging Face Transformers
- Parameter compatibility designed for T5X checkpoint conversion