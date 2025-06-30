#!/usr/bin/env python3
"""
Test script to verify TrackMyPDB installation
@author Anu Gamage - Licensed under MIT License
"""

import sys
import importlib

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - FAILED: {e}")
        return False

def main():
    """Main testing function"""
    print("🧬 TrackMyPDB Installation Test")
    print("=" * 40)
    
    # Test core dependencies
    modules = [
        'streamlit',
        'pandas',
        'numpy',
        'requests',
        'tqdm',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    success_count = 0
    
    for module in modules:
        if test_import(module):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(modules)} modules imported successfully")
    
    # Test RDKit separately (most likely to fail)
    print("\n🧪 Testing RDKit (cheminformatics)...")
    if test_import('rdkit'):
        print("✅ RDKit is available")
    else:
        print("❌ RDKit not available")
        print("   Install with: pip install rdkit-pypi")
        print("   Or with conda: conda install -c conda-forge rdkit")
    
    # Test backend modules
    print("\n🔧 Testing backend modules...")
    sys.path.append('backend')
    
    try:
        from backend.heteroatom_extractor import HeteroatomExtractor
        print("✅ HeteroatomExtractor - OK")
    except ImportError as e:
        print(f"❌ HeteroatomExtractor - FAILED: {e}")
    
    try:
        from backend.similarity_analyzer import MolecularSimilarityAnalyzer
        print("✅ MolecularSimilarityAnalyzer - OK")
    except ImportError as e:
        print(f"❌ MolecularSimilarityAnalyzer - FAILED: {e}")
    
    print("\n🎯 Installation test completed!")
    print("If all tests passed, run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 