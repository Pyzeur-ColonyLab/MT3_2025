# YourMT3 Integration pour Brev

## 🎯 Objectif

Remplacer MT3 (0 notes à cause du vocabulaire mismatch) par YourMT3+ qui fonctionne.

## 📊 Contexte

- **MT3 actuel**: Génère uniquement token 1133 → 0 notes
- **Root cause**: Vocabulaire mismatch (1536 vs 1491 tokens)
- **Solution**: YourMT3+ (2024) avec checkpoint pré-entraîné

## 🚀 Setup Rapide sur Brev

```bash
# 1. Clone le repo
git clone <repo-url>
cd MT3

# 2. Login HuggingFace
huggingface-cli login
# Entrer votre token HF

# 3. Run setup script
bash setup_yourmt3_brev.sh

# 4. Upload audio (si pas déjà fait)
# Uploader 02.HowardShore-TheShire.flac sur Brev

# 5. Test transcription
python3 test_yourmt3.py
```

## ⏱️ Temps Estimé

- **Setup**: ~5 minutes
- **Download checkpoint**: ~2 minutes (536 MB)
- **Transcription**: ~2-3 minutes sur A10G GPU

## 📁 Fichiers

```
MT3/
├── setup_yourmt3_brev.sh        # Setup automatique
├── test_yourmt3.py              # Script de test
├── YOURMT3_SETUP.md             # Documentation complète
├── YOURMT3_ANALYSIS.md          # Analyse technique détaillée
└── README_YOURMT3.md            # Ce fichier
```

## ✅ Résultat Attendu

```
============================================================
YourMT3 Test: The Shire Transcription
============================================================

1. Loading YourMT3 model...
   Device: cuda
   Model: YPTF.MoE+Multi (noPS)
   ✅ Model loaded successfully

2. Loading audio: 02.HowardShore-TheShire.flac
   Duration: 149.48s

3. Transcribing with YourMT3...
   Converting audio... 0.5s
   Model inference... 120s
   Post processing... 1.2s

✅ Transcription complete!
   MIDI file: ./model_output/TheShire_YourMT3.mid
   Total notes: XXX (> 0 attendu!)
   Instruments: 13

   ✅ SUCCESS: XXX notes detected!
```

## 🔍 Vérification

Comparer avec MT3:
```bash
# MT3 (broken)
python3 test_real_music.py
# → 0 notes ❌

# YourMT3 (working)
python3 test_yourmt3.py
# → > 0 notes ✅
```

## 📚 Documentation

- **Setup complet**: `YOURMT3_SETUP.md`
- **Analyse technique**: `YOURMT3_ANALYSIS.md`
- **Paper YourMT3**: https://arxiv.org/abs/2407.04822
- **Demo HF**: https://huggingface.co/spaces/mimbres/YourMT3

## ⚠️ Notes

- **Checkpoint**: 536 MB (téléchargé automatiquement)
- **Auth HF**: Requise pour download checkpoint
- **GPU recommandé**: A10G ou mieux
- **Mémoire**: ~4-6 GB VRAM en inference
