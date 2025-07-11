import pytest
from backend.nl_interface import NaturalLanguageInterface
from backend.agent_core import TrackMyPDBAgent, AgentQuery
import pandas as pd
import datetime

@pytest.fixture
def agent():
    return TrackMyPDBAgent()

@pytest.fixture
def nl_interface(agent):
    return NaturalLanguageInterface(agent)

@pytest.mark.asyncio
async def test_process_user_input(nl_interface):
    # Test basic query processing
    query = "Extract heteroatoms from P53"
    await nl_interface._process_user_input(query)
    assert len(nl_interface.agent.query_history) > 0

def test_display_results(nl_interface):
    # Test results display with sample data
    test_results = {
        "heteroatom_results": pd.DataFrame({
            "atom": ["N", "O", "S"],
            "count": [10, 15, 5]
        })
    }
    nl_interface._display_results(test_results, "heteroatom_analysis")
    # Visual verification required in Streamlit