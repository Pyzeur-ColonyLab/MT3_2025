# T5X to PyTorch Conversion Guide

Complete step-by-step guide to convert MT3 T5X checkpoints (JAX/Flax) to PyTorch format.

---

## Prerequisites

### Download T5X Checkpoints

**Official MT3 Checkpoints** are available on Google Cloud Storage:

```bash
# Create checkpoints directory
mkdir -p checkpoints/mt3

# Download using gsutil (recommended)
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/* checkpoints/mt3/

# Alternative: Download specific checkpoint
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/
```

**If you don't have gsutil:**

```bash
# Install Google Cloud SDK
# Ubuntu/Debian:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Install gsutil
pip install gsutil
```

**Alternative sources:**

1. **Hugging Face Mirror** (if available):
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/SungBeom/mt3
   # Checkpoints are in: mt3/checkpoints/
   ```

2. **Direct download** (specific versions):
   - Check https://github.com/magenta/mt3 for direct download links
   - Look in the README or releases section

3. **Magenta website**:
   - Visit https://magenta.tensorflow.org/datasets/mt3
   - Follow download instructions

### Checkpoint Sizes

- **Full MT3 model**: ~180-200 MB (compressed)
- **After extraction**: ~500 MB - 1 GB (Zarr format with 147 parameter folders)

### Required Files
- T5X checkpoint directory (Zarr format with multiple `target.*` folders)
- Conversion script: `convert_t5x_to_pytorch.py`

### Python Environment
```bash
# Create virtual environment
python3 -m venv mt3_env
source mt3_env/bin/activate  # On Windows: mt3_env\Scripts\activate

# Install dependencies
pip install torch numpy zarr
```

---

## Step 1: Verify Checkpoint Structure

Your T5X checkpoint should have this structure:

```
checkpoints/mt3/
├── checkpoint              # Metadata file
├── target.encoder.layers_0.attention.query.kernel/
│   ├── .zarray            # Zarr metadata (JSON)
│   └── 0.0                # Compressed data (gzip)
├── target.encoder.layers_0.attention.key.kernel/
│   ├── .zarray
│   └── 0.0
├── target.decoder.layers_0.self_attention.query.kernel/
│   ├── .zarray
│   └── 0.0
└── ... (147 parameter folders total)
```

### Verify Zarr Format
```bash
# Check one parameter folder
ls -la checkpoints/mt3/target.encoder.layers_0.attention.query.kernel/

# Should show:
# .zarray (JSON metadata file)
# 0.0 (compressed data file)

# Inspect metadata
cat checkpoints/mt3/target.encoder.layers_0.attention.query.kernel/.zarray

# Should show JSON like:
# {
#   "chunks": [512, 384],
#   "compressor": {"id": "gzip", "level": 1},
#   "dtype": "<f4",
#   "shape": [512, 384],
#   ...
# }
```

---

## Step 2: Run Conversion Script

### Basic Usage

```bash
python convert_t5x_to_pytorch.py <t5x_checkpoint_dir> [output_dir]
```

### Example

```bash
python convert_t5x_to_pytorch.py \
    /home/ubuntu/mt3-pytorch/checkpoints/mt3 \
    ./pretrained/
```

### Expected Output

```
======================================================================
🔄 CONVERSION T5X (JAX/Zarr) → PYTORCH
======================================================================
🔍 Chargement du checkpoint T5X (format Zarr)...
📊 147 dossiers de paramètres trouvés
  ✓ target.decoder.layers_0.encoder_decoder_attention.key.kernel
    Shape: (512, 384), Dtype: float32
  ✓ target.decoder.layers_0.encoder_decoder_attention.out.kernel
    Shape: (384, 512), Dtype: float32
  ...
  ... et 137 autres chargés

✅ 147/147 paramètres chargés avec succès

🔄 Conversion des noms de paramètres...
  target.decoder.layers_0.encoder_decoder_attention.key.kernel
    → decoder.layer.0.cross_attn.k_proj.weight
       Shape: [384, 512]
  ...
  ... et 137 autres

✅ 147 paramètres convertis

💾 Sauvegarde du checkpoint PyTorch...
✅ Checkpoint sauvegardé: pretrained/mt3_converted.pth
   Taille: 183.55 MB
   Paramètres: 45,875,200

📝 Mapping sauvegardé: pretrained/parameter_mapping.txt
📝 Création de config.json...
✅ Config sauvegardée: pretrained/config.json
   d_model: 1536
   vocab_size: 512
   encoder_layers: 8
   decoder_layers: 8

======================================================================
✅ CONVERSION TERMINÉE !
======================================================================

Fichiers créés dans: pretrained
  - mt3_converted.pth          (poids du modèle)
  - config.json                (configuration)
  - parameter_mapping.txt      (liste des paramètres)
```

---

## Step 3: Verify Generated Files

### Check Output Files

```bash
ls -lh pretrained/

# Should show:
# mt3_converted.pth       (183.55 MB)
# config.json             (few KB)
# parameter_mapping.txt   (text file)
```

### Inspect Config

```bash
cat pretrained/config.json
```

**IMPORTANT:** The auto-detected config may have swapped values. Verify and correct:

```json
{
  "model_type": "mt3",
  "architecture": "t5",
  "vocab_size": 1536,      // ← Should be 1536
  "d_model": 512,          // ← Should be 512
  "d_ff": 2048,            // ← Should be 2048 (4 * d_model)
  "num_encoder_layers": 8,
  "num_decoder_layers": 8,
  "num_heads": 8,
  "dropout_rate": 0.1,
  "layer_norm_epsilon": 1e-6,
  "converted_from": "t5x_checkpoint_zarr"
}
```

### Verify Parameter Mapping

```bash
head -30 pretrained/parameter_mapping.txt

# Should show parameter names and shapes:
# decoder.layer.0.cross_attn.k_proj.weight
#   Shape: [384, 512]
#   Dtype: torch.float32
#   Numel: 196,608
```

---

## Step 4: Validate Conversion

### Python Validation Script

```python
import torch
import json

# Load checkpoint
checkpoint = torch.load('pretrained/mt3_converted.pth')
state_dict = checkpoint['model_state_dict']
metadata = checkpoint['metadata']

# Check metadata
print("Checkpoint Metadata:")
print(f"  Source: {metadata['source']}")
print(f"  Parameters: {metadata['num_parameters']}")
print(f"  Total params: {metadata['total_params']:,}")

# Check some key parameters
key_params = [
    'encoder.layer.0.attn.q_proj.weight',
    'decoder.layer.0.cross_attn.k_proj.weight',
    'decoder.layer.0.ffn.fc1.weight'
]

print("\nKey Parameters Check:")
for param_name in key_params:
    if param_name in state_dict:
        tensor = state_dict[param_name]
        print(f"✓ {param_name}: {list(tensor.shape)}")
    else:
        print(f"✗ {param_name}: NOT FOUND")

# Load config
with open('pretrained/config.json', 'r') as f:
    config = json.load(f)

print("\nConfiguration:")
print(f"  d_model: {config['d_model']}")
print(f"  vocab_size: {config['vocab_size']}")
print(f"  encoder_layers: {config['num_encoder_layers']}")
print(f"  decoder_layers: {config['num_decoder_layers']}")
```

---

## Troubleshooting

### Error: "zarr not installed"

```bash
pip install zarr
```

### Error: "buffer size must be a multiple of element size"

This means the zarr library is needed to handle gzip compression:

```bash
pip install zarr numcodecs
```

### Error: "No structure could be identified"

Your checkpoint format might be different. Verify:

```bash
# Check if target.* folders exist
ls checkpoints/mt3/ | grep target | head -5

# Check if .zarray files exist
find checkpoints/mt3/ -name ".zarray" | head -5
```

### Config values are swapped

Manually edit `pretrained/config.json`:
- Swap `vocab_size` and `d_model` if needed
- Correct `d_ff` (should be `4 * d_model` for standard T5)

### Missing parameters

Check `parameter_mapping.txt` to see which parameters were successfully converted. Some T5X checkpoints may have different naming conventions.

---

## Understanding the Conversion

### Key Transformations

1. **Format**: Zarr (compressed directories) → PyTorch .pth file
2. **Names**: T5X naming → PyTorch naming convention
3. **Transpose**: JAX (in, out) → PyTorch (out, in) for linear layers

### Name Mapping Examples

| T5X Name | PyTorch Name |
|----------|--------------|
| `target.encoder.layers_0.attention.query.kernel` | `encoder.layer.0.attn.q_proj.weight` |
| `target.decoder.layers_0.self_attention.key.kernel` | `decoder.layer.0.self_attn.k_proj.weight` |
| `target.decoder.layers_0.mlp.wi_0.kernel` | `decoder.layer.0.ffn.fc1.weight` |
| `target.encoder.continuous_inputs_projection.kernel` | `encoder.input_projection.weight` |

### Weight Shapes

**Attention Projections** (with 8 heads):
- Query/Key/Value: `[384, 512]` = `[d_model * 3/4, d_model]`
- Output: `[512, 384]`

**Feed-Forward Network**:
- FC1: `[2048, 512]` = `[d_ff, d_model]`
- FC2 (wo): `[512, 2048]` = `[d_model, d_ff]`

---

## Next Steps

After successful conversion, you have:

1. **mt3_converted.pth**: PyTorch checkpoint ready to load
2. **config.json**: Model configuration
3. **parameter_mapping.txt**: Complete parameter list (for debugging)

### Using the Checkpoint

```python
import torch
from your_model import MT3Model

# Load config
config = json.load(open('pretrained/config.json'))

# Create model
model = MT3Model(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    num_heads=config['num_heads'],
    d_ff=config['d_ff']
)

# Load weights
checkpoint = torch.load('pretrained/mt3_converted.pth')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
```

### Required Implementations

Before you can use this checkpoint, you need to implement:

1. **MT3Model class**: PyTorch model architecture matching the parameter names
2. **Vocabulary**: Token-to-MIDI event mapping
3. **Audio preprocessing**: Convert audio to model inputs
4. **Token decoding**: Convert model outputs to MIDI files

See `MT3_IMPLEMENTATION_ROADMAP.md` for detailed implementation guide.

---

## Summary Checklist

- [ ] T5X checkpoint in Zarr format verified
- [ ] `zarr` library installed
- [ ] Conversion script executed successfully
- [ ] All 147 parameters converted
- [ ] `mt3_converted.pth` generated (~183 MB)
- [ ] `config.json` values verified and corrected if needed
- [ ] `parameter_mapping.txt` reviewed
- [ ] Checkpoint validation script passed
- [ ] Ready to implement MT3Model

---

## Additional Resources

- Original MT3 repository: https://github.com/magenta/mt3
- T5 architecture: https://arxiv.org/abs/1910.10683
- Zarr format: https://zarr.readthedocs.io/
- PyTorch model conversion: https://pytorch.org/docs/stable/notes/serialization.html
