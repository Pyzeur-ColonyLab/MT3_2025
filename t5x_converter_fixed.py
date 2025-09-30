#!/usr/bin/env python3
"""
Convertisseur de checkpoints T5X (JAX/Flax) vers PyTorch pour MT3
Version optimisée pour le format Zarr/Orbax
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("⚠️  zarr non installé, installation recommandée: pip install zarr")


class T5XToPyTorchConverter:
    """Convertit les checkpoints T5X vers PyTorch"""
    
    def __init__(self, t5x_checkpoint_dir, output_dir):
        self.t5x_checkpoint_dir = Path(t5x_checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_zarr_array(self, param_dir):
        """Charge un array depuis un dossier Zarr avec support de compression"""
        
        # Méthode 1: Utiliser zarr si disponible (recommandé)
        if ZARR_AVAILABLE:
            try:
                z = zarr.open(str(param_dir), mode='r')
                return np.array(z)
            except Exception as e:
                print(f"    ⚠️  Zarr lib échoué: {e}, tentative manuelle...")
        
        # Méthode 2: Lecture manuelle avec décompression
        zarray_file = param_dir / '.zarray'
        
        if not zarray_file.exists():
            return None
        
        try:
            # Lire les métadonnées Zarr
            with open(zarray_file, 'r') as f:
                metadata = json.load(f)
            
            shape = tuple(metadata['shape'])
            dtype = np.dtype(metadata['dtype'])
            order = metadata.get('order', 'C')
            compressor = metadata.get('compressor', None)
            
            # Charger le chunk
            chunk_files = sorted([f for f in param_dir.iterdir() 
                                 if f.is_file() and f.name != '.zarray'])
            
            if not chunk_files:
                return None
            
            chunk_file = chunk_files[0]
            
            with open(chunk_file, 'rb') as f:
                compressed_data = f.read()
            
            # Décompresser si nécessaire
            if compressor is not None:
                try:
                    import numcodecs
                    codec = numcodecs.get_codec(compressor)
                    data_bytes = codec.decode(compressed_data)
                except ImportError:
                    import gzip
                    # Fallback sur gzip standard
                    try:
                        data_bytes = gzip.decompress(compressed_data)
                    except:
                        print(f"    ⚠️  Impossible de décompresser (installez: pip install zarr)")
                        return None
            else:
                data_bytes = compressed_data
            
            # Convertir en array numpy
            data = np.frombuffer(data_bytes, dtype=dtype)
            
            # Reshape
            if len(shape) > 0:
                data = data.reshape(shape, order=order)
            
            return data
                
        except Exception as e:
            print(f"    ⚠️  Erreur: {e}")
            return None
    
    def load_t5x_checkpoint(self):
        """Charge le checkpoint T5X depuis les dossiers Zarr"""
        print("🔍 Chargement du checkpoint T5X (format Zarr)...")
        
        if not ZARR_AVAILABLE:
            print("\n❌ ERREUR: La bibliothèque 'zarr' est requise !")
            print("   Installation: pip install zarr")
            print("   Puis relancez le script.\n")
            return None
        
        params = {}
        
        # Lister tous les dossiers target.*
        param_dirs = [d for d in self.t5x_checkpoint_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('target.')]
        
        if not param_dirs:
            print("❌ Aucun dossier de paramètres trouvé")
            return None
        
        print(f"📊 {len(param_dirs)} dossiers de paramètres trouvés")
        
        loaded_count = 0
        failed_count = 0
        
        for param_dir in sorted(param_dirs):
            param_name = param_dir.name
            
            # Charger le tenseur Zarr
            data = self.load_zarr_array(param_dir)
            
            if data is not None:
                params[param_name] = data
                loaded_count += 1
                
                if loaded_count <= 10:
                    print(f"  ✓ {param_name}")
                    print(f"    Shape: {data.shape}, Dtype: {data.dtype}")
            else:
                failed_count += 1
                if failed_count <= 3:
                    print(f"  ✗ {param_name}: échec de chargement")
        
        if loaded_count > 10:
            print(f"  ... et {loaded_count - 10} autres chargés")
        
        if failed_count > 0:
            print(f"  ⚠️  {failed_count} paramètres n'ont pas pu être chargés")
        
        print(f"\n✅ {loaded_count}/{len(param_dirs)} paramètres chargés avec succès")
        
        return params if params else None
    
    def map_t5x_to_pytorch(self, t5x_weights):
        """Mappe les noms T5X vers PyTorch"""
        print("\n🔄 Conversion des noms de paramètres...")
        
        pytorch_state = OrderedDict()
        
        # Mapping des noms T5X → PyTorch
        name_mappings = [
            # Retirer le préfixe target.
            ('target.', ''),
            
            # Structure des couches
            ('encoder.layers_', 'encoder.layer.'),
            ('decoder.layers_', 'decoder.layer.'),
            
            # Attention
            ('self_attention.', 'self_attn.'),
            ('encoder_decoder_attention.', 'cross_attn.'),
            ('.attention.', '.attn.'),
            
            # Projections d'attention
            ('query.kernel', 'q_proj.weight'),
            ('key.kernel', 'k_proj.weight'),
            ('value.kernel', 'v_proj.weight'),
            ('out.kernel', 'out_proj.weight'),
            
            # FFN/MLP
            ('mlp.wi_0.kernel', 'ffn.fc1.weight'),
            ('mlp.wi_1.kernel', 'ffn.fc2.weight'),
            ('mlp.wo.kernel', 'ffn.fc_out.weight'),
            
            # Embeddings
            ('token_embedder.embedding', 'embed_tokens.weight'),
            ('continuous_inputs_projection.kernel', 'input_projection.weight'),
            ('logits_dense.kernel', 'lm_head.weight'),
            
            # Layer norm
            ('.layer_norm.scale', '.norm.weight'),
            ('.final_layer_norm.scale', '.final_norm.weight'),
            
            # Fallback pour kernel → weight
            ('.kernel', '.weight'),
            ('.scale', '.weight'),
        ]
        
        converted_count = 0
        skipped = []
        
        for t5x_name, weight in t5x_weights.items():
            # Appliquer les mappings
            pytorch_name = t5x_name
            for t5x_pattern, pt_pattern in name_mappings:
                pytorch_name = pytorch_name.replace(t5x_pattern, pt_pattern)
            
            try:
                # Convertir en numpy float32
                weight_np = np.array(weight, dtype=np.float32)
                
                # Transposer les matrices linéaires
                # T5X: (in_features, out_features)
                # PyTorch: (out_features, in_features)
                if '.weight' in pytorch_name and len(weight_np.shape) == 2:
                    weight_np = weight_np.T
                
                # Convertir en tensor PyTorch
                pytorch_state[pytorch_name] = torch.from_numpy(weight_np)
                converted_count += 1
                
                if converted_count <= 10:
                    print(f"  {t5x_name}")
                    print(f"    → {pytorch_name}")
                    print(f"       Shape: {list(weight_np.shape)}")
                        
            except Exception as e:
                skipped.append((t5x_name, str(e)))
                if len(skipped) <= 3:
                    print(f"  ⚠️  Erreur avec {t5x_name}: {e}")
        
        if converted_count > 10:
            print(f"  ... et {converted_count - 10} autres")
        
        if skipped and len(skipped) > 3:
            print(f"\n⚠️  {len(skipped)} paramètres ignorés")
        
        print(f"\n✅ {converted_count} paramètres convertis")
        return pytorch_state
    
    def save_pytorch_checkpoint(self, pytorch_state, model_name="mt3_converted"):
        """Sauvegarde au format PyTorch"""
        print(f"\n💾 Sauvegarde du checkpoint PyTorch...")
        
        output_file = self.output_dir / f"{model_name}.pth"
        
        checkpoint = {
            'model_state_dict': pytorch_state,
            'metadata': {
                'source': 'T5X/JAX checkpoint (Zarr format)',
                'converter': 't5x_to_pytorch_zarr',
                'num_parameters': len(pytorch_state),
                'total_params': sum(p.numel() for p in pytorch_state.values())
            }
        }
        
        try:
            torch.save(checkpoint, output_file)
            size_mb = output_file.stat().st_size / 1e6
            total_params = sum(p.numel() for p in pytorch_state.values())
            
            print(f"✅ Checkpoint sauvegardé: {output_file}")
            print(f"   Taille: {size_mb:.2f} MB")
            print(f"   Paramètres: {total_params:,}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
    
    def save_parameter_mapping(self, pytorch_state):
        """Sauvegarde la liste des paramètres pour debug"""
        mapping_file = self.output_dir / "parameter_mapping.txt"
        
        with open(mapping_file, 'w') as f:
            f.write("LISTE DES PARAMÈTRES CONVERTIS\n")
            f.write("=" * 70 + "\n\n")
            
            for name, tensor in pytorch_state.items():
                f.write(f"{name}\n")
                f.write(f"  Shape: {list(tensor.shape)}\n")
                f.write(f"  Dtype: {tensor.dtype}\n")
                f.write(f"  Numel: {tensor.numel():,}\n\n")
        
        print(f"📝 Mapping sauvegardé: {mapping_file}")
    
    def create_config(self, pytorch_state):
        """Crée la configuration PyTorch basée sur les paramètres"""
        print("\n📝 Création de config.json...")
        
        # Détecter les dimensions depuis les paramètres
        # Chercher un embedding pour trouver d_model
        d_model = 512  # défaut
        vocab_size = 1536  # défaut
        
        for name, tensor in pytorch_state.items():
            if 'embed_tokens.weight' in name:
                vocab_size, d_model = tensor.shape
                break
            elif 'input_projection.weight' in name:
                d_model = tensor.shape[0]
        
        # Compter les couches
        num_encoder_layers = 0
        num_decoder_layers = 0
        
        for name in pytorch_state.keys():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                num_encoder_layers = max(num_encoder_layers, layer_num + 1)
            elif 'decoder.layer.' in name:
                layer_num = int(name.split('decoder.layer.')[1].split('.')[0])
                num_decoder_layers = max(num_decoder_layers, layer_num + 1)
        
        config = {
            "model_type": "mt3",
            "architecture": "t5",
            "vocab_size": int(vocab_size),
            "d_model": int(d_model),
            "d_ff": int(d_model * 4),  # Standard T5
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": 8,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "converted_from": "t5x_checkpoint_zarr"
        }
        
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config sauvegardée: {config_file}")
        print(f"   d_model: {config['d_model']}")
        print(f"   vocab_size: {config['vocab_size']}")
        print(f"   encoder_layers: {config['num_encoder_layers']}")
        print(f"   decoder_layers: {config['num_decoder_layers']}")
    
    def convert(self):
        """Exécute la conversion complète"""
        print("=" * 70)
        print("🔄 CONVERSION T5X (JAX/Zarr) → PYTORCH")
        print("=" * 70)
        
        # 1. Charger T5X
        t5x_weights = self.load_t5x_checkpoint()
        if not t5x_weights:
            print("\n❌ Impossible de charger le checkpoint")
            return False
        
        # 2. Convertir
        pytorch_state = self.map_t5x_to_pytorch(t5x_weights)
        
        if len(pytorch_state) == 0:
            print("\n❌ Aucun paramètre converti!")
            return False
        
        # 3. Sauvegarder
        output_file = self.save_pytorch_checkpoint(pytorch_state)
        if not output_file:
            return False
        
        # 4. Sauvegarder le mapping (debug)
        self.save_parameter_mapping(pytorch_state)
        
        # 5. Config
        self.create_config(pytorch_state)
        
        print("\n" + "=" * 70)
        print("✅ CONVERSION TERMINÉE !")
        print("=" * 70)
        print(f"\nFichiers créés dans: {self.output_dir}")
        print("  - mt3_converted.pth          (poids du modèle)")
        print("  - config.json                (configuration)")
        print("  - parameter_mapping.txt      (liste des paramètres)")
        
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_t5x_to_pytorch.py <t5x_checkpoint_dir> [output_dir]")
        print("\nExemple:")
        print("  python convert_t5x_to_pytorch.py ./checkpoints/mt3/ ./pretrained/")
        print("\nNote: Ce script lit les checkpoints Zarr (format avec .zarray)")
        sys.exit(1)
    
    t5x_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './pretrained'
    
    if not os.path.exists(t5x_dir):
        print(f"❌ Dossier introuvable: {t5x_dir}")
        sys.exit(1)
    
    converter = T5XToPyTorchConverter(t5x_dir, output_dir)
    success = converter.convert()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
