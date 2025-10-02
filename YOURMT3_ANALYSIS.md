# YourMT3 Analysis et Décision d'Intégration

Date: 2025-10-01
Status: Analysis Complete - Décision Requise

## 📊 Résumé Exécutif

**Problème actuel**: MT3 avec checkpoint T5X converti génère uniquement le token 1133 → 0 notes
**Root cause**: Vocabulaire mismatch (Model: 1536 tokens vs Decoder: 1491 tokens)
**Alternative**: YourMT3+ (2024) démo HF fonctionne "ok (5/10)"

## 🔍 Analyse YourMT3+

### Architecture

**YourMT3 != MT3**: Complexité significativement plus élevée

```
YourMT3 (2024)                    vs        MT3 (2021)
├─ Encoder                                  ├─ Encoder
│  ├─ T5-small (option)                    │  └─ T5-small
│  ├─ Perceiver-TF (démo)                  │
│  └─ Conformer (option)                   │
├─ Decoder                                  ├─ Decoder
│  ├─ Multi-channel T5 (démo)              │  └─ T5-small standard
│  └─ Single-channel T5                    │
├─ Mixture of Experts (MoE)                 └─ (pas de MoE)
├─ Multi-instrument channels
└─ Lightning + custom task manager
```

### Dépendances

**YourMT3 Requirements**:
```
pytorch-lightning>=2.2.1
transformers==4.45.1
torch, torchaudio
librosa, einops
mido
```

**Notre code actuel**:
```
torch, torchaudio
pretty_midi (pour MIDI)
librosa (pour audio)
```

### Fichiers Clés

```
yourmt3_space/
├── amt/src/
│   ├── model/
│   │   ├── ymt3.py                 # 800+ lignes, PyTorch Lightning
│   │   ├── t5mod.py                # T5 modifié
│   │   ├── perceiver_mod.py        # Perceiver-TF encoder
│   │   ├── conformer_mod.py        # Conformer encoder
│   │   └── lm_head.py              # Classification head
│   ├── config/
│   │   ├── config.py               # Configs audio/model/shared
│   │   ├── task.py                 # Task manager
│   │   └── vocabulary.py           # Vocabulaires
│   └── utils/
│       ├── task_manager.py         # Tokenization/detokenization
│       ├── note2event.py           # Note → Event conversion
│       └── event2note.py           # Event → Note conversion
├── model_helper.py                  # Load model + inference
└── requirements.txt
```

### Modèle Utilisé dans la Démo

**Checkpoint**: `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt`

**Configuration**:
- Encoder: Perceiver-TF + Mixture of Experts (2/8)
- Decoder: Multi-channel T5-small
- Tokenizer: MT3 tokens + extension Singing
- Audio: Spec (pas melspec), hop=300, 512 mel bins
- Training: Cross-dataset stem augmentation, no pitch-shifting
- Precision: BF16-mixed training, FP16 inference

### Pipeline d'Inférence

```python
# 1. Model Loading (model_helper.py:25-123)
model = YourMT3(
    audio_cfg=audio_cfg,
    model_cfg=model_cfg,
    shared_cfg=shared_cfg,
    task_manager=tm,
    ...
).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# 2. Audio Preprocessing (model_helper.py:131-136)
audio, sr = torchaudio.load(audio_file)
audio = torch.mean(audio, dim=0).unsqueeze(0)  # Mono
audio = torchaudio.functional.resample(audio, sr, 16000)
audio_segments = slice_padded_array(audio, input_frames, input_frames)
audio_segments = torch.from_numpy(audio_segments).to(device).unsqueeze(1)

# 3. Model Inference (model_helper.py:141)
pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)

# 4. Post-processing (model_helper.py:151-158)
for ch in range(num_channels):
    zipped_note_events_and_tie = task_manager.detokenize_list_batches(pred_token_arr_ch)
    pred_notes_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
    pred_notes_in_file.append(pred_notes_ch)
pred_notes = mix_notes(pred_notes_in_file)

# 5. Write MIDI (model_helper.py:161-162)
write_model_output_as_midi(pred_notes, output_dir, track_name, midi_inverse_vocab)
```

## 🎯 Options d'Intégration

### Option A: Intégration Complète YourMT3

**Approche**: Copier `amt/src/` entièrement dans notre projet

**Avantages**:
- ✅ Accès à l'architecture complète
- ✅ Support multi-instruments
- ✅ Qualité prouvée (démo fonctionne)

**Inconvénients**:
- ❌ **~50 fichiers Python à intégrer**
- ❌ Dépendances lourdes (PyTorch Lightning, etc.)
- ❌ Complexité élevée (800+ lignes pour le modèle seul)
- ❌ Maintenance difficile
- ❌ Nécessite téléchargement checkpoint (~500MB)

**Effort**: 🔴 3-5 jours

### Option B: Wrapper Simplifié YourMT3

**Approche**: Créer `yourmt3_inference.py` qui appelle le code existant

**Avantages**:
- ✅ Réutilise le code YourMT3 tel quel
- ✅ Moins de duplication
- ✅ Mise à jour facile

**Inconvénients**:
- ❌ Dépend toujours de `amt/src/`
- ❌ Nécessite configuration complexe
- ❌ Pas de contrôle sur l'architecture

**Effort**: 🟡 1-2 jours

### Option C: Checkpoint MT3 Officiel depuis HuggingFace

**Approche**: Chercher checkpoint MT3 pré-entraîné sur HF Hub

**Status**:
- Google bucket vide: ❌ `gs://magentadata/models/mt3/checkpoints`
- HuggingFace: 🔍 À vérifier

**Avantages**:
- ✅ Notre code MT3 déjà fonctionnel (sauf checkpoint)
- ✅ Architecture simple et comprise
- ✅ Pas de dépendances lourdes

**Inconvénients**:
- ⚠️ Checkpoint officiel peut ne pas exister sur HF
- ⚠️ Qualité possiblement inférieure à YourMT3

**Effort**: 🟢 <1 jour SI checkpoint existe

### Option D: Basic-Pitch (Alternative Lightweight)

**Approche**: Utiliser Spotify Basic-Pitch au lieu de MT3/YourMT3

**Model**: `spotify/basic-pitch` sur HuggingFace

**Avantages**:
- ✅ Architecture simple (CNN + RNN)
- ✅ API HF transformers standard
- ✅ Léger et rapide
- ✅ Maintenance active

**Inconvénients**:
- ⚠️ Moins précis que MT3/YourMT3
- ⚠️ Pas multi-instruments natif
- ⚠️ Architecture différente (pas T5)

**Effort**: 🟢 <1 jour

## 🧪 Tests Effectués

### MT3 Converti (Notre Code)
```
Checkpoint: mt3_converted_fixed.pth
Audio: 02.HowardShore-TheShire.flac (149s)
Résultat:
  ❌ Generated 2048 tokens
  ❌ Unique tokens: 3 (only [0, 1133, 1133...])
  ❌ Notes detected: 0
  ❌ MIDI file: vide
Root cause: Vocabulary mismatch (1536 vs 1491 tokens)
```

### YourMT3 Demo HF
```
Model: YPTF.MoE+Multi (noPS)
Interface: https://huggingface.co/spaces/mimbres/YourMT3
Résultat:
  ✅ Fonctionne "ok (5/10)"
  ✅ Génère des notes
  ✅ MIDI jouable
Limitation: YouTube bloqué, exemples pré-transcrits seulement
```

## 📊 Comparaison Quantitative

| Critère | MT3 (notre code) | YourMT3+ Full | YourMT3 Wrapper | Basic-Pitch |
|---------|------------------|---------------|-----------------|-------------|
| **Effort** | 🟢 Code existe | 🔴 3-5 jours | 🟡 1-2 jours | 🟢 <1 jour |
| **Complexité** | 🟢 Simple | 🔴 Très élevée | 🟡 Moyenne | 🟢 Simple |
| **Qualité** | ❌ 0 notes | ✅ 5/10 | ✅ 5/10 | ⚠️ 3-4/10 |
| **Maintenance** | 🟢 Facile | 🔴 Difficile | 🟡 Moyenne | 🟢 Facile |
| **Dépendances** | 🟢 Légères | 🔴 Lourdes | 🟡 Moyennes | 🟢 Légères |
| **Multi-instruments** | ⚠️ Possible | ✅ Natif | ✅ Natif | ❌ Limité |
| **Checkpoint** | ❌ Invalide | ✅ Existe | ✅ Existe | ✅ HF Hub |

## 💡 Recommandation

### Stratégie Hybride (Option C → B → D)

**Phase 1: Vérifier HF Hub pour MT3 (30 min)**
```bash
# Chercher checkpoint MT3 officiel
huggingface-cli search "MT3" --filter "model"
huggingface-cli search "music transcription" --filter "model"
```
- SI trouvé → Utiliser avec notre code existant ✅
- SI pas trouvé → Phase 2

**Phase 2: YourMT3 Wrapper Minimal (1-2 jours)**
- Copier `amt/src/` dans `external/yourmt3/`
- Créer `yourmt3_inference.py` wrapper simple
- Télécharger checkpoint depuis HF Space
- Tester avec "The Shire"
- SI qualité acceptable (≥4/10) → DONE ✅
- SI qualité insuffisante → Phase 3

**Phase 3: Fallback Basic-Pitch (1 jour)**
- Implémenter `basic_pitch_inference.py`
- Comparer avec YourMT3
- Choisir le meilleur compromis qualité/complexité

## 📝 Actions Immédiates

1. ✅ Clone YourMT3 Space: **DONE**
2. ✅ Analyse architecture: **DONE**
3. ⏳ **NEXT**: Rechercher checkpoint MT3 sur HF Hub
4. ⏳ Si pas trouvé: Créer wrapper YourMT3
5. ⏳ Tester avec "The Shire"
6. ⏳ Documenter résultats

## 🔗 Références

- **YourMT3 Paper**: [arxiv.org/abs/2407.04822](https://arxiv.org/abs/2407.04822)
- **YourMT3 HF Space**: [huggingface.co/spaces/mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3)
- **YourMT3 Model**: [huggingface.co/mimbres/YourMT3](https://huggingface.co/mimbres/YourMT3)
- **YourMT3 GitHub**: [github.com/mimbres/YourMT3](https://github.com/mimbres/YourMT3) (empty repo)
- **MT3 Original**: Google Magenta (bucket vide)
- **Basic-Pitch**: [huggingface.co/spotify/basic-pitch](https://huggingface.co/spotify/basic-pitch)

---

## ⚠️ Décision Requise

**Question pour l'utilisateur**:

Quelle approche préférez-vous ?

1. **Option Rapide** 🟢: Rechercher MT3 sur HF Hub (30 min) puis Basic-Pitch si échec
2. **Option Qualité** 🟡: YourMT3 Wrapper (~2 jours) pour 5/10 qualité
3. **Option Complète** 🔴: Intégration YourMT3 full (~5 jours) pour contrôle total

**Ma recommandation**: Option Rapide → Qualité (stratégie hybride)
