#!/usr/bin/env python3
"""
Test script to verify TrackMyPDB installation
@author Anu Gamage - Licensed under MIT License
"""

import sys
import importlib
import pytest

@pytest.mark.parametrize("module_name", [
    "streamlit",
    "pandas",
    "numpy",
    "requests",
    "tqdm",
    "matplotlib",
    "seaborn",
    "plotly",
    "rdkit"
])
def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        assert True, f"{module_name} imported successfully"
    except ImportError:
        pytest.fail(f"Failed to import {module_name}")

def test_backend_imports():
    """Test backend module imports"""
    try:
        # Test heteroatom extractor
        from backend.heteroatom_extractor import OptimizedHeteroatomExtractor
        from backend.similarity_analyzer import SimilarityAnalyzer
        from backend.agent_core import TrackMyPDBAgent
        from backend.nl_interface import NaturalLanguageInterface
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import backend modules: {e}")