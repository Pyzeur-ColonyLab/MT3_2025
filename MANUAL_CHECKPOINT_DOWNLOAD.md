# Manual MT3 Checkpoint Download Guide

If the automated download fails, follow these step-by-step instructions.

---

## 🎯 Goal

Download the MT3 T5X checkpoint and convert it to PyTorch format to get `mt3_converted.pth` (~183 MB).

---

## ✅ Option 1: Download Pre-Converted Checkpoint (Easiest)

If someone has already converted the checkpoint, you can download it directly:

### Check These Sources:

1. **Kunato's MT3 PyTorch Repository**
   ```bash
   # Check releases for pre-converted checkpoint
   https://github.com/kunato/mt3-pytorch/releases
   ```

2. **Hugging Face Models**
   ```bash
   # Search for MT3 PyTorch checkpoints
   https://huggingface.co/models?search=mt3
   ```

3. **Google Drive / Dropbox Shares**
   - Check MT3 GitHub issues for community-shared links
   - Search "mt3_converted.pth download" on Google

If you find a pre-converted `mt3_converted.pth`:
```bash
# Download to your MT3_2025 directory
wget <direct_link> -O mt3_converted.pth

# Verify it loaded correctly
python3 -c "import torch; c=torch.load('mt3_converted.pth'); print('✓ Valid checkpoint:', c['metadata']['total_params'], 'parameters')"
```

**Then skip to verification section at the bottom!**

---

## 📥 Option 2: Download T5X and Convert (Manual)

If you need to download the original T5X checkpoint:

### Step 1: Install Google Cloud SDK

```bash
# Download and install
curl https://sdk.cloud.google.com | bash

# Restart shell
exec -l $SHELL

# Initialize (no Google account needed for public data)
gcloud init --skip-diagnostics
```

When prompted:
- **"Pick configuration to use"**: Choose [1] Re-initialize
- **"Choose account"**: Choose "Log in with a new account" or skip
- **"Pick cloud project"**: Press Enter to skip (not needed for public data)

### Step 2: Download with gsutil

```bash
# Create directory
mkdir -p checkpoints/mt3

# Download checkpoint (~200-500 MB)
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/

# This may take 5-10 minutes
```

**Note:** If you get authentication errors, try:
```bash
# Set anonymous access
gcloud config set auth/disable_credentials true

# Try download again
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/
```

### Step 3: Verify Download

```bash
# Check parameter folders
ls checkpoints/mt3/mt3/ | grep target | wc -l
# Should show: 147

# Check one parameter
ls checkpoints/mt3/mt3/target.encoder.layers_0.attention.query.kernel/
# Should show: .zarray  0.0
```

### Step 4: Convert to PyTorch

```bash
python3 t5x_converter_fixed.py checkpoints/mt3/mt3 .
```

Expected output:
```
🔄 CONVERSION T5X (JAX/Zarr) → PYTORCH
...
✅ Checkpoint sauvegardé: ./mt3_converted.pth
   Taille: 183.55 MB
   Paramètres: 45,875,200
```

---

## 📥 Option 3: Alternative Sources

### A. Hugging Face Spaces

```bash
# Install git-lfs
sudo apt-get install git-lfs
git lfs install

# Clone repository with checkpoint
git clone https://huggingface.co/spaces/SungBeom/mt3

# Check for checkpoint
ls mt3/checkpoints/

# If found, copy and convert
cp -r mt3/checkpoints/mt3 checkpoints/
python3 t5x_converter_fixed.py checkpoints/mt3 .
```

### B. Direct HTTP Download (if available)

```bash
# Try direct download from Google Storage
wget -r -np -nH --cut-dirs=4 \
  https://storage.googleapis.com/magentadata/models/mt3/checkpoints/mt3/ \
  -P checkpoints/mt3/

# Convert
python3 t5x_converter_fixed.py checkpoints/mt3/mt3 .
```

### C. Kaggle Datasets

1. Search https://www.kaggle.com/datasets for "mt3 checkpoint"
2. Download if available
3. Extract and convert

### D. Contact Repository Maintainers

Open an issue on:
- https://github.com/magenta/mt3
- https://github.com/kunato/mt3-pytorch

Ask if there's a direct download link for the checkpoint.

---

## ✅ Verification

Once you have `mt3_converted.pth`:

### Check File Size
```bash
ls -lh mt3_converted.pth
# Should be ~183 MB (170-190 MB range is OK)
```

### Verify Contents
```bash
python3 << 'EOF'
import torch

checkpoint = torch.load('mt3_converted.pth')
metadata = checkpoint['metadata']

print("✓ Checkpoint loaded successfully")
print(f"✓ Total parameters: {metadata['total_params']:,}")
print(f"✓ Parameter count: {metadata['num_parameters']}")
print(f"✓ Source: {metadata['source']}")

# Check key parameters exist
state_dict = checkpoint['model_state_dict']
key_params = [
    'encoder.layer.0',
    'decoder.layer.0',
]

found = sum(1 for name in state_dict.keys() if any(kp in name for kp in key_params))
print(f"✓ Found {found} encoder/decoder parameters")

if found > 0:
    print("\n✅ Checkpoint is valid!")
else:
    print("\n⚠️  Warning: Checkpoint structure may be incorrect")
EOF
```

Expected output:
```
✓ Checkpoint loaded successfully
✓ Total parameters: 45,875,200
✓ Parameter count: 147
✓ Source: T5X/JAX checkpoint (Zarr format)
✓ Found 147 encoder/decoder parameters

✅ Checkpoint is valid!
```

### Test Import
```bash
python3 -c "from inference import MT3Inference; print('✓ MT3 imports successful')"
```

---

## 🚀 Next Steps

Once you have a valid `mt3_converted.pth`:

### Quick Test
```bash
# Create a short test audio file or download one
# Example: download a piano recording

# Test transcription
python example_inference.py test.wav --checkpoint mt3_converted.pth --output test.mid

# If successful, you'll see:
# ✅ Transcribed X notes
# 📝 Saved to: test.mid
```

### Use Jupyter Notebook
```bash
jupyter notebook MT3_Test_Notebook.ipynb
```

---

## ❓ Troubleshooting

### "gsutil: command not found"
```bash
pip install gsutil
# OR install full SDK:
curl https://sdk.cloud.google.com | bash
```

### "Access Denied" from gsutil
```bash
# Try anonymous access
gcloud config set auth/disable_credentials true
gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/
```

### "No module named 'zarr'"
```bash
pip install zarr numcodecs
```

### Download is very slow
- Try during off-peak hours
- Use a different network connection
- Try alternative sources (Hugging Face, direct HTTP)

### Can't find any working download source

**Temporary Solution:**
1. Open an issue on https://github.com/Pyzeur-ColonyLab/MT3_2025/issues
2. Ask if someone can share their converted checkpoint
3. Or ask for help with gsutil authentication

**Community Help:**
- MT3 Discord/Slack channels
- r/MachineLearning on Reddit
- Magenta discussion forums

---

## 📧 Need Help?

If none of these methods work:

1. **Check the original MT3 repository:**
   https://github.com/magenta/mt3/issues

2. **Search for "mt3 checkpoint download" issues:**
   - GitHub issues
   - Stack Overflow
   - Reddit r/MachineLearning

3. **Alternative: Use a different model:**
   - Basic Pitch: https://github.com/spotify/basic-pitch
   - Omnizart: https://github.com/Music-and-Culture-Technology-Lab/omnizart

---

## 📊 Checkpoint Information

For reference, the correct checkpoint should have:

- **File size**: ~183 MB for PyTorch format
- **Parameters**: 45,875,200 total
- **Architecture**: T5 encoder-decoder
- **Format**: PyTorch .pth file
- **Keys**: `model_state_dict` and `metadata`

If your checkpoint matches these specs, it should work!