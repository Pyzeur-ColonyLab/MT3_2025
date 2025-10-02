# YourMT3 Setup et Utilisation

## ✅ Étapes Complétées

1. ✅ Clone du Space YourMT3
2. ✅ Authentification HuggingFace
3. ✅ Téléchargement du checkpoint (536 MB)
4. ✅ Script de test créé

## 📦 Installation des Dépendances

### Option 1: Environnement Virtuel (Recommandé)

```bash
cd /Volumes/T7/Dyapason/instrument-recognition-app/MT3

# Créer environnement virtuel
python3 -m venv venv_yourmt3

# Activer
source venv_yourmt3/bin/activate

# Installer dépendances YourMT3
pip install torch torchaudio
pip install pytorch-lightning>=2.2.1
pip install transformers==4.45.1
pip install librosa einops mido
pip install pretty_midi  # Pour analyser les résultats

# Installer dépendances YourMT3 complètes
pip install -r yourmt3_space/requirements.txt
```

### Option 2: Installation Globale

```bash
pip3 install torch torchaudio pytorch-lightning transformers==4.45.1 librosa einops mido pretty_midi
```

## 🚀 Exécution du Test

```bash
cd /Volumes/T7/Dyapason/instrument-recognition-app/MT3

# Avec environnement virtuel
source venv_yourmt3/bin/activate
python test_yourmt3.py

# Ou directement
python3 test_yourmt3.py
```

## 📁 Structure des Fichiers

```
MT3/
├── test_yourmt3.py                  # Script de test (CRÉÉ)
├── 02.HowardShore-TheShire.flac     # Audio source (149s)
├── yourmt3_space/                   # Clone du Space HF
│   ├── amt/src/                     # Code source YourMT3
│   │   ├── model/
│   │   │   ├── ymt3.py              # Modèle principal
│   │   │   └── ...
│   │   ├── config/
│   │   └── utils/
│   ├── amt/logs/2024/
│   │   └── mc13_256_g4.../checkpoints/
│   │       └── last.ckpt            # Checkpoint (536 MB) ✅
│   ├── model_helper.py              # Helpers chargement/inférence
│   └── requirements.txt
└── model_output/                    # MIDI output (sera créé)
    └── TheShire_YourMT3.mid
```

## 🎯 Résultat Attendu

**Si succès**:
```
============================================================
YourMT3 Test: The Shire Transcription
============================================================

1. Loading YourMT3 model...
   Device: cuda (ou cpu)
   Model: YPTF.MoE+Multi (noPS)
   ✅ Model loaded successfully

2. Loading audio: 02.HowardShore-TheShire.flac
   Duration: 149.48s
   Sample rate: 44100 Hz
   Channels: 2

3. Transcribing with YourMT3...
   Converting audio... X.XXs
   Model inference... X.XXs
   Post processing... X.XXs

✅ Transcription complete!
   MIDI file: ./model_output/TheShire_YourMT3.mid
   File size: XXXXX bytes
   Total notes: XXX (devrait être > 0!)
   Instruments: X

   ✅ SUCCESS: XXX notes detected!

============================================================
YourMT3 test complete
============================================================
```

**Comparaison avec MT3 converti**:
- MT3: **0 notes** (vocabulaire mismatch)
- YourMT3: **> 0 notes** attendu (qualité 5/10 selon démo)

## 🔍 Dépannage

### Erreur: `ModuleNotFoundError: No module named 'torch'`
→ Installer les dépendances (voir ci-dessus)

### Erreur: `No such file or directory: yourmt3_space/amt/src`
→ Vérifier que vous êtes dans le bon répertoire: `cd /Volumes/T7/Dyapason/instrument-recognition-app/MT3`

### Erreur: Checkpoint not found
→ Vérifier le checkpoint:
```bash
ls -lh yourmt3_space/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt
# Devrait montrer 536M
```

### Mémoire insuffisante
→ Réduire batch size dans model_helper.py ligne 141:
```python
pred_token_arr, _ = model.inference_file(bsz=4, ...)  # au lieu de bsz=8
```

## 📊 Prochaines Étapes

1. **Exécuter le test** → Voir si YourMT3 génère des notes
2. **Analyser le MIDI** → Comparer avec démo HF
3. **Décider**:
   - Si qualité acceptable (≥4/10) → Intégrer YourMT3
   - Si qualité insuffisante → Tester Basic-Pitch
4. **Nettoyer** → Supprimer fichiers temporaires MT3

## 🎵 Notes Techniques

**Configuration Modèle**:
- Encoder: Perceiver-TF + MoE (8 experts, top-2)
- Decoder: Multi-channel T5 (13 instruments)
- Audio: Spectrogram (512 bins), hop=300
- Tokenizer: MT3 + Singing extension
- Inference: BF16/FP16 precision

**Temps d'exécution estimé**:
- CPU: ~10-15 minutes pour 149s audio
- GPU (CUDA): ~2-3 minutes
