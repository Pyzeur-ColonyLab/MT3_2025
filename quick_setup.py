#!/usr/bin/env python3
"""
Quick MT3 Checkpoint Setup
Tries multiple download methods automatically
"""

import os
import sys
import subprocess
import urllib.request
import json
from pathlib import Path

def print_status(msg, emoji="ℹ"):
    print(f"{emoji}  {msg}")

def print_success(msg):
    print(f"✓  {msg}")

def print_error(msg):
    print(f"✗  {msg}")

def check_command(cmd):
    """Check if command exists"""
    return subprocess.run(['which', cmd], capture_output=True).returncode == 0

def download_huggingface():
    """Try downloading from Hugging Face"""
    print_status("Method 1: Trying Hugging Face Hub...", "📥")

    # Check git-lfs
    if not check_command('git-lfs'):
        print_status("Installing git-lfs...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'git-lfs'],
                      capture_output=True)
        subprocess.run(['git', 'lfs', 'install'], capture_output=True)

    # Clone repo
    print_status("Cloning repository...")
    result = subprocess.run(
        ['git', 'clone', 'https://huggingface.co/spaces/SungBeom/mt3', 'temp_mt3_hf'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        # Check for checkpoint
        checkpoint_path = Path('temp_mt3_hf/checkpoints/mt3')
        if checkpoint_path.exists():
            print_status("Found checkpoint, copying...")
            os.makedirs('checkpoints/mt3', exist_ok=True)
            subprocess.run(['cp', '-r', str(checkpoint_path), 'checkpoints/'])
            subprocess.run(['rm', '-rf', 'temp_mt3_hf'])
            print_success("Downloaded from Hugging Face")
            return True

    subprocess.run(['rm', '-rf', 'temp_mt3_hf'], capture_output=True)
    print_error("Hugging Face download failed")
    return False

def download_direct_http():
    """Try direct HTTP download"""
    print_status("Method 2: Trying direct HTTP...", "📥")

    urls = [
        "https://storage.googleapis.com/magentadata/models/mt3/checkpoints/mt3/",
    ]

    for url in urls:
        print_status(f"Trying: {url}")
        result = subprocess.run(
            ['wget', '-q', '--spider', url],
            capture_output=True
        )

        if result.returncode == 0:
            print_status("URL accessible, downloading...")
            result = subprocess.run([
                'wget', '-r', '-np', '-nH', '--cut-dirs=4',
                '-R', 'index.html*', url, '-P', 'checkpoints/mt3/'
            ])

            if result.returncode == 0:
                print_success("Downloaded via HTTP")
                return True

    print_error("Direct HTTP download failed")
    return False

def setup_gcloud():
    """Try to setup gcloud and download"""
    print_status("Method 3: Setting up Google Cloud...", "☁️")

    # Check if gcloud exists
    if not check_command('gcloud'):
        print_status("Installing Google Cloud SDK...")
        print_status("This requires manual steps:")
        print("")
        print("  1. Run: curl https://sdk.cloud.google.com | bash")
        print("  2. Restart your shell: exec -l $SHELL")
        print("  3. Run: gcloud init --skip-diagnostics")
        print("  4. Re-run this script")
        print("")
        return False

    # Try anonymous access
    print_status("Attempting anonymous download...")
    subprocess.run(['gcloud', 'config', 'set', 'auth/disable_credentials', 'true'],
                  capture_output=True)

    result = subprocess.run([
        'gsutil', '-m', 'cp', '-r',
        'gs://magentadata/models/mt3/checkpoints/mt3/',
        'checkpoints/mt3/'
    ])

    if result.returncode == 0:
        print_success("Downloaded via Google Cloud")
        return True

    print_error("Google Cloud download failed")
    return False

def convert_checkpoint():
    """Run conversion script"""
    print("")
    print_status("Converting T5X checkpoint to PyTorch...", "🔄")

    result = subprocess.run([
        'python3', 't5x_converter_fixed.py',
        'checkpoints/mt3/mt3', '.'
    ])

    return result.returncode == 0

def verify_checkpoint():
    """Verify the converted checkpoint"""
    print("")
    print_status("Verifying checkpoint...", "✅")

    try:
        import torch
        checkpoint = torch.load('mt3_converted.pth')
        metadata = checkpoint['metadata']

        print_success(f"Checkpoint loaded successfully")
        print_success(f"Total parameters: {metadata['total_params']:,}")

        return True
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False

def show_manual_instructions():
    """Show manual download instructions"""
    print("")
    print("="*70)
    print_error("AUTOMATIC DOWNLOAD FAILED")
    print("="*70)
    print("")
    print_status("Please try manual download methods:")
    print("")
    print("📖 Read the detailed guide:")
    print("   cat MANUAL_CHECKPOINT_DOWNLOAD.md")
    print("")
    print("🔧 Quick manual methods:")
    print("")
    print("  1. Setup Google Cloud authentication:")
    print("     curl https://sdk.cloud.google.com | bash")
    print("     exec -l $SHELL")
    print("     gcloud init --skip-diagnostics")
    print("     gsutil -m cp -r gs://magentadata/models/mt3/checkpoints/mt3/ checkpoints/mt3/")
    print("")
    print("  2. Check for pre-converted checkpoints:")
    print("     https://github.com/kunato/mt3-pytorch/releases")
    print("     https://huggingface.co/models?search=mt3")
    print("")
    print("  3. Ask the community:")
    print("     https://github.com/magenta/mt3/issues")
    print("")

def main():
    print("="*70)
    print("🎵 MT3 Checkpoint Quick Setup")
    print("="*70)
    print("")

    # Check if checkpoint already exists
    if os.path.exists('mt3_converted.pth'):
        print_status("mt3_converted.pth already exists!")
        response = input("Recreate it? (y/N): ")
        if response.lower() != 'y':
            print_status("Keeping existing checkpoint.")
            return 0

    # Install Python dependencies
    print_status("Installing Python dependencies...")
    subprocess.run(['pip', 'install', '-q', 'zarr', 'numcodecs'],
                  capture_output=True)

    # Try download methods in order
    download_success = False

    if download_huggingface():
        download_success = True
    elif download_direct_http():
        download_success = True
    elif setup_gcloud():
        download_success = True

    if not download_success:
        show_manual_instructions()
        return 1

    # Convert checkpoint
    if convert_checkpoint():
        if verify_checkpoint():
            print("")
            print("="*70)
            print_success("MT3 CHECKPOINT SETUP COMPLETE!")
            print("="*70)
            print("")
            print_status("Next steps:")
            print("  1. Test: python example_inference.py audio.wav --checkpoint mt3_converted.pth")
            print("  2. Or use: jupyter notebook MT3_Test_Notebook.ipynb")
            print("")
            return 0

    print_error("Setup failed")
    return 1

if __name__ == "__main__":
    sys.exit(main())