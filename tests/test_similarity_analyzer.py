import pytest
import pandas as pd
import numpy as np
from backend.similarity_analyzer import SimilarityAnalyzer
from unittest.mock import MagicMock, patch

@pytest.fixture
def analyzer():
    return SimilarityAnalyzer()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'PDB_ID': ['1ABC', '2DEF', '3GHI'],
        'Heteroatom_Code': ['LIG1', 'LIG2', 'LIG3'],
        'SMILES': ['CC(=O)O', 'CCO', 'CCC'],  # Simple SMILES for testing
        'Chemical_Name': ['Acetic acid', 'Ethanol', 'Propane'],
        'Formula': ['C2H4O2', 'C2H6O', 'C3H8']
    })

def test_init(analyzer):
    assert hasattr(analyzer, 'fingerprints')
    assert hasattr(analyzer, 'valid_molecules')
    assert isinstance(analyzer.fingerprints, dict)
    assert isinstance(analyzer.valid_molecules, dict)

def test_load_and_process_dataframe(analyzer, sample_data):
    result = analyzer.load_and_process_dataframe(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_data)  # Should have same or fewer rows
    assert 'Fingerprint' in result.columns
    
    # Test storage in analyzer
    assert len(analyzer.fingerprints) > 0
    assert len(analyzer.valid_molecules) > 0

def test_find_similar_ligands(analyzer, sample_data):
    processed_df = analyzer.load_and_process_dataframe(sample_data)
    target_smiles = 'CC(=O)O'  # Acetic acid
    
    result = analyzer.find_similar_ligands(
        target_smiles=target_smiles,
        processed_df=processed_df,
        top_n=2,
        min_similarity=0.0
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 2  # Should respect top_n parameter
    assert 'Tanimoto_Similarity' in result.columns
    assert all(0 <= sim <= 1 for sim in result['Tanimoto_Similarity'])

@patch('streamlit.columns')
@patch('streamlit.markdown')
@patch('streamlit.subheader')
def test_display_similarity_results(mock_subheader, mock_markdown, mock_columns, analyzer, sample_data):
    # Mock streamlit functions
    col1, col2, col3, col4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    # Make columns return different numbers of columns based on argument
    mock_columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    
    processed_df = analyzer.load_and_process_dataframe(sample_data)
    similarity_results = analyzer.find_similar_ligands('CC(=O)O', processed_df)
    display_df = analyzer.display_similarity_results('CC(=O)O', similarity_results)
    
    assert isinstance(display_df, pd.DataFrame)
    if len(display_df) > 0:
        assert 'PDB_ID' in display_df.columns
        assert 'Tanimoto_Similarity' in display_df.columns
        
    # Verify Streamlit functions were called
    mock_markdown.assert_called()
    mock_subheader.assert_called()
    mock_columns.assert_called()