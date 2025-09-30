# MT3 Checkpoint Setup Guide

Complete guide to download and convert the MT3 checkpoint on your Brev NVIDIA instance.

## 🚀 Quick Start

### Option A: Automated Setup (Try First)

```bash
# Method 1: Main setup script
bash setup_mt3_checkpoint.sh
```

**If this fails (common due to gsutil authentication)**, try:

```bash
# Method 2: Python-based alternative (tries multiple sources)
python3 quick_setup.py

# Method 3: Alternative bash script
bash download_checkpoint_alternative.sh
```

### Option B: Manual Download (If automated fails)

See the detailed **[MANUAL_CHECKPOINT_DOWNLOAD.md](MANUAL_CHECKPOINT_DOWNLOAD.md)** guide.

Quick summary:
1. **Pre-converted checkpoint**: Check https://github.com/kunato/mt3-pytorch/releases
2. **Setup gcloud**: Follow authentication steps in manual guide
3. **Alternative sources**: Hugging Face, direct HTTP download

---

## 📦 What You'll Get

**Expected output files:**
- `mt3_converted.pth` (~183 MB) - PyTorch checkpoint
- `config.json` - Model configuration
- `parameter_mapping.txt` - Parameter list (for debugging)

**Setup process:**
1. ✅ Check for existing checkpoint
2. ✅ Install required dependencies (zarr, numcodecs)
3. ✅ Download T5X checkpoint from Google Cloud (~200-500 MB)
4. ✅ Convert to PyTorch format
5. ✅ Verify the conversion

---

## 📋 Prerequisites

### System Requirements
- **Internet connection** for downloading checkpoint
- **Disk space**: ~1 GB free (500 MB for T5X + 200 MB for PyTorch)
- **RAM**: 4 GB minimum
- **GPU**: NVIDIA GPU with CUDA (already available on Brev)

### Python Dependencies
```bash
pip install zarr numcodecs gsutil
```

These are installed automatically by the setup script.

---

## 🔧 Manual Setup (Advanced)

If you prefer manual control or the automated script fails:

### Step 1: Install gsutil

```bash
# Install gsutil
pip install gsutil

# Verify installation
gsutil version
```

### Step 2: Download T5X Checkpoint

```bash
# Create checkpoint directory
mkdir -p checkpoints/mt3

# Download from Google Cloud Storage
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/

# This will download ~200-500 MB of data (147 parameter folders)
```

**Download time estimate:**
- Fast connection (100+ Mbps): 2-5 minutes
- Medium connection (10-50 Mbps): 5-10 minutes
- Slow connection (<10 Mbps): 10-20 minutes

### Step 3: Verify Checkpoint Structure

```bash
# Check parameter folders
ls checkpoints/mt3/mt3/ | grep target | wc -l
# Should show: 147

# Verify Zarr format
ls checkpoints/mt3/mt3/target.encoder.layers_0.attention.query.kernel/
# Should show: .zarray and 0.0 (or similar chunk files)
```

### Step 4: Convert to PyTorch

```bash
python3 t5x_converter_fixed.py checkpoints/mt3/mt3 .
```

**Expected output:**
```
======================================================================
🔄 CONVERSION T5X (JAX/Zarr) → PYTORCH
======================================================================
🔍 Chargement du checkpoint T5X (format Zarr)...
📊 147 dossiers de paramètres trouvés
  ✓ target.encoder.layers_0.attention.query.kernel
    Shape: (512, 384), Dtype: float32
  ...
✅ 147/147 paramètres chargés avec succès

🔄 Conversion des noms de paramètres...
  ...
✅ 147 paramètres convertis

💾 Sauvegarde du checkpoint PyTorch...
✅ Checkpoint sauvegardé: ./mt3_converted.pth
   Taille: 183.55 MB
   Paramètres: 45,875,200

======================================================================
✅ CONVERSION TERMINÉE !
======================================================================
```

---

## ✅ Verification

### Verify Checkpoint File

```bash
# Check file exists and size
ls -lh mt3_converted.pth
# Should be ~183 MB

# Load and inspect with Python
python3 << EOF
import torch
checkpoint = torch.load('mt3_converted.pth')
print(f"✓ Checkpoint loaded successfully")
print(f"✓ Total parameters: {checkpoint['metadata']['total_params']:,}")
print(f"✓ Parameter count: {checkpoint['metadata']['num_parameters']}")
EOF
```

**Expected output:**
```
✓ Checkpoint loaded successfully
✓ Total parameters: 45,875,200
✓ Parameter count: 147
```

### Verify Configuration

```bash
cat config.json
```

**Should show:**
```json
{
  "model_type": "mt3",
  "architecture": "t5",
  "vocab_size": 1536,
  "d_model": 512,
  "d_ff": 2048,
  "num_encoder_layers": 8,
  "num_decoder_layers": 8,
  "num_heads": 8,
  "dropout_rate": 0.1,
  "layer_norm_epsilon": 1e-6,
  "converted_from": "t5x_checkpoint_zarr"
}
```

### Test with MT3

```bash
# Quick import test
python3 -c "from inference import MT3Inference; print('✓ MT3 imports successful')"

# If you have an audio file
python example_inference.py test.wav --checkpoint mt3_converted.pth --output test.mid
```

---

## 🐛 Troubleshooting

### Error: "gsutil: command not found"

```bash
# Install gsutil
pip install gsutil

# Or install full Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Error: "No module named 'zarr'"

```bash
pip install zarr numcodecs
```

### Error: "Unable to register cuFFT factory" (warnings only)

These warnings are normal and don't affect functionality. They come from TensorFlow/JAX dependencies and can be ignored.

### Download is very slow

**Alternative download sources:**

1. **Hugging Face Mirror:**
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/SungBeom/mt3
   # Checkpoint will be in: mt3/checkpoints/
   ```

2. **Direct HTTP download** (if available):
   Check https://github.com/magenta/mt3 for direct download links

### Conversion fails with "buffer size must be a multiple of element size"

Install the full zarr stack:
```bash
pip install zarr numcodecs
```

### Config values look wrong

The conversion script auto-detects configuration from parameter shapes. If values seem swapped:

```bash
# Manually edit config.json
nano config.json

# Ensure:
# - vocab_size: 1536
# - d_model: 512
# - d_ff: 2048 (4 * d_model)
```

### Checkpoint is incomplete (<100 parameters)

The download may have been interrupted. Delete and re-download:

```bash
rm -rf checkpoints/mt3
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/
```

---

## 📊 Checkpoint Details

### File Structure

```
MT3_2025/
├── mt3_converted.pth          # PyTorch checkpoint (183 MB)
├── config.json                # Model configuration
├── parameter_mapping.txt      # Parameter list (debugging)
├── t5x_converter_fixed.py     # Conversion script
├── setup_mt3_checkpoint.sh    # Automated setup script
└── checkpoints/               # Temporary (can be deleted after conversion)
    └── mt3/
        └── mt3/
            ├── target.encoder.layers_0.*.kernel/
            ├── target.decoder.layers_0.*.kernel/
            └── ... (147 parameter folders)
```

### Model Architecture

- **Type**: T5 encoder-decoder
- **Parameters**: 45.8 million
- **Layers**: 8 encoder + 8 decoder
- **Hidden size**: 512 (d_model)
- **Feed-forward**: 2048 (d_ff)
- **Attention heads**: 8
- **Vocabulary**: 1536 tokens

### Parameter Breakdown

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Encoder | ~22.9M | 50% |
| Decoder | ~22.9M | 50% |
| **Total** | **45.8M** | **100%** |

---

## 🎯 Next Steps

Once you have `mt3_converted.pth`:

### 1. Test Basic Transcription

```bash
# With a short audio file (<30 seconds)
python example_inference.py audio.wav --checkpoint mt3_converted.pth
```

### 2. Use Jupyter Notebook

```bash
jupyter notebook MT3_Test_Notebook.ipynb
```

The notebook provides:
- ✅ Dependency verification
- ✅ GPU checking
- ✅ Multiple transcription strategies
- ✅ Long audio processing
- ✅ Batch processing
- ✅ Performance benchmarking

### 3. Integrate into Your Pipeline

```python
from inference import MT3Inference

# Initialize
inference = MT3Inference(
    checkpoint_path="mt3_converted.pth",
    device="cuda"
)

# Transcribe
result = inference.transcribe(
    audio_path="song.wav",
    output_path="song.mid"
)

print(f"Transcribed {result['num_notes']} notes")
```

---

## 📚 Additional Resources

- **Original MT3 Paper**: https://arxiv.org/abs/2111.03017
- **MT3 Repository**: https://github.com/magenta/mt3
- **Checkpoint Source**: gs://magentadata/models/mt3/checkpoints/
- **Conversion Guide**: See `t5x_conversion_guide.md` for detailed technical information

---

## ❓ FAQ

### Q: How long does the setup take?
**A:** 10-15 minutes total (5-10 min download + 2-5 min conversion)

### Q: Can I use CPU instead of GPU?
**A:** Yes, but transcription will be much slower (~10-20x). The checkpoint itself works on CPU.

### Q: Do I need to keep the checkpoints/ directory?
**A:** No, once you have `mt3_converted.pth`, you can delete `checkpoints/` to save ~500 MB.

### Q: Can I share the converted checkpoint?
**A:** The checkpoint is derived from Google's MT3 model under Apache 2.0 license. Check license terms before redistribution.

### Q: The script says zarr is not installed but I installed it
**A:** Try: `pip install --upgrade zarr numcodecs`

### Q: Can I convert other T5X checkpoints?
**A:** Yes! The `t5x_converter_fixed.py` script works with any T5X checkpoint in Zarr format.

---

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review `t5x_conversion_guide.md` for detailed technical information
3. Verify your Python environment: `pip list | grep -E "(torch|zarr)"`
4. Check the conversion script output for specific error messages

For persistent issues, check:
- MT3 GitHub issues: https://github.com/magenta/mt3/issues
- Ensure you're on a Brev instance with proper GPU access