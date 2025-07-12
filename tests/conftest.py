import os
import sys
import pytest
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock streamlit
mock_streamlit = MagicMock()
mock_streamlit.session_state = MagicMock()
mock_streamlit.session_state.chat_history = []
sys.modules['streamlit'] = mock_streamlit

from backend.heteroatom_extractor import OptimizedHeteroatomExtractor
from backend.similarity_analyzer import SimilarityAnalyzer
from backend.agent_core import TrackMyPDBAgent
from backend.nl_interface import NaturalLanguageInterface

@pytest.fixture
def agent():
    return TrackMyPDBAgent()

@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return {
        'uniprot_ids': ['P11511', 'P56524'],
        'pdb_ids': ['3EQM', '4F2W'],
        'heteroatom_codes': ['ASD', 'BCA']
    }

@pytest.fixture
def heteroatom_extractor():
    """Create OptimizedHeteroatomExtractor instance for testing"""
    return OptimizedHeteroatomExtractor()

@pytest.fixture
def similarity_analyzer():
    return SimilarityAnalyzer()

@pytest.fixture
def nl_interface(agent):
    return NaturalLanguageInterface(agent)