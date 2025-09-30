#!/usr/bin/env python3
"""
Syntax and Structure Validation for MT3 Model

Validates the Python syntax and imports of the MT3 model implementation
without requiring PyTorch to be installed.
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path: Path) -> dict:
    """Validate Python syntax for a file."""
    results = {
        'file': str(file_path),
        'syntax_valid': False,
        'imports_valid': False,
        'classes_found': [],
        'functions_found': [],
        'errors': []
    }

    try:
        # Read file
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse syntax
        tree = ast.parse(content, filename=str(file_path))
        results['syntax_valid'] = True

        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                results['classes_found'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                results['functions_found'].append(node.name)

        # Check imports (basic validation)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        results['imports'] = imports
        results['imports_valid'] = True

    except SyntaxError as e:
        results['errors'].append(f"Syntax error: {e}")
    except Exception as e:
        results['errors'].append(f"Error: {e}")

    return results


def validate_model_structure():
    """Validate the overall model structure."""
    models_dir = Path('.')

    # Files to validate
    files_to_check = [
        'mt3_model.py',
        'checkpoint_utils.py',
        'validate_model.py',
        '__init__.py'
    ]

    print("🔍 Validating MT3 Model Implementation")
    print("=" * 50)

    all_valid = True

    for file_name in files_to_check:
        file_path = models_dir / file_name

        if not file_path.exists():
            print(f"❌ {file_name}: File not found")
            all_valid = False
            continue

        results = validate_python_syntax(file_path)

        if results['syntax_valid']:
            print(f"✅ {file_name}: Syntax valid")
            print(f"   Classes: {len(results['classes_found'])}")
            print(f"   Functions: {len(results['functions_found'])}")
        else:
            print(f"❌ {file_name}: Syntax errors")
            for error in results['errors']:
                print(f"   {error}")
            all_valid = False

    print("\n📊 Expected Implementation Components:")
    print("-" * 30)

    # Check mt3_model.py specifically
    mt3_model_path = models_dir / 'mt3_model.py'
    if mt3_model_path.exists():
        results = validate_python_syntax(mt3_model_path)

        expected_classes = [
            'MT3Config', 'MT3Model', 'MT3Encoder', 'MT3Decoder',
            'RMSNorm', 'MT3Attention', 'MT3Block', 'MT3BlockDecoder'
        ]

        expected_functions = [
            'create_mt3_model', 'forward', 'generate'
        ]

        print(f"Expected classes: {len(expected_classes)}")
        print(f"Found classes: {len(results['classes_found'])}")

        for cls in expected_classes:
            if cls in results['classes_found']:
                print(f"  ✅ {cls}")
            else:
                print(f"  ❌ {cls} (missing)")
                all_valid = False

        print(f"\nKey functions present:")
        for func in expected_functions:
            if func in results['functions_found']:
                print(f"  ✅ {func}")
            else:
                print(f"  ❌ {func} (missing)")

    print("\n🎯 Model Specifications Validation:")
    print("-" * 30)

    # Read the configuration from file
    try:
        with open(mt3_model_path, 'r') as f:
            content = f.read()

        # Check for expected parameter values
        specs_found = {
            'vocab_size: int = 1536': 'vocab_size: int = 1536' in content,
            'd_model: int = 512': 'd_model: int = 512' in content,
            'num_encoder_layers: int = 8': 'num_encoder_layers: int = 8' in content,
            'num_decoder_layers: int = 8': 'num_decoder_layers: int = 8' in content,
            'num_heads: int = 8': 'num_heads: int = 8' in content,
            'd_ff: int = 1024': 'd_ff: int = 1024' in content,
        }

        for spec, found in specs_found.items():
            if found:
                print(f"  ✅ {spec}")
            else:
                print(f"  ❌ {spec} (not found)")
                all_valid = False

    except Exception as e:
        print(f"  ❌ Could not validate specifications: {e}")
        all_valid = False

    print("\n📋 Architecture Requirements:")
    print("-" * 30)

    architecture_checks = [
        ('T5-style relative position bias', 'relative_attention_bias' in content),
        ('Shared embedding layer', 'self.shared = nn.Embedding' in content),
        ('Cross-attention in decoder', 'MT3LayerCrossAttention' in content),
        ('RMSNorm layer normalization', 'class RMSNorm' in content),
        ('Parameter initialization', 'init_weights' in content),
        ('Generate method', 'def generate(' in content),
    ]

    for feature, check in architecture_checks:
        if check:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} (missing)")
            all_valid = False

    print(f"\n{'='*50}")
    if all_valid:
        print("🎉 All validation checks passed!")
        print("✨ MT3Model implementation appears complete and correct")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Run full validation: python validate_model.py")
        print("3. Test with actual checkpoint loading")
        return True
    else:
        print("💥 Some validation checks failed!")
        print("🔧 Please review and fix the issues above")
        return False


if __name__ == "__main__":
    success = validate_model_structure()
    sys.exit(0 if success else 1)