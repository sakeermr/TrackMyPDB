import pytest
from backend.agent_core import TrackMyPDBAgent, AgentQuery
import datetime

@pytest.mark.asyncio
async def test_process_query(agent):
    query = AgentQuery(
        text="Extract heteroatoms from 1ABC",
        timestamp=datetime.datetime.now(),
        query_type="heteroatom_analysis"
    )
    
    result = await agent.process_query(query)
    assert result is not None
    assert isinstance(result, dict)
    assert 'query_type' in result

def test_query_history(agent):
    query = AgentQuery(
        text="Test query",
        timestamp=datetime.datetime.now(),
        query_type="test"
    )
    agent.add_to_history(query)
    
    assert len(agent.query_history) > 0
    assert agent.query_history[-1].text == "Test query"

def test_validate_query():
    # Test query validation
    valid_query = AgentQuery(
        text="Valid query",
        timestamp=datetime.datetime.now(),
        query_type="heteroatom_analysis"
    )
    assert TrackMyPDBAgent.validate_query(valid_query)

    invalid_query = AgentQuery(
        text="",
        timestamp=datetime.datetime.now(),
        query_type="invalid_type"
    )
    with pytest.raises(ValueError):
        TrackMyPDBAgent.validate_query(invalid_query)