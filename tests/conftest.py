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

from backend.agent_core import TrackMyPDBAgent
from backend.heteroatom_extractor import HeteroatomExtractor
from backend.similarity_analyzer import SimilarityAnalyzer
from backend.nl_interface import NaturalLanguageInterface

@pytest.fixture
def agent():
    return TrackMyPDBAgent()

@pytest.fixture
def heteroatom_extractor():
    return HeteroatomExtractor()

@pytest.fixture
def similarity_analyzer():
    return SimilarityAnalyzer()

@pytest.fixture
def nl_interface(agent):
    return NaturalLanguageInterface(agent)