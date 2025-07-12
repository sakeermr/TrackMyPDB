import pytest
from backend.heteroatom_extractor import OptimizedHeteroatomExtractor
import pandas as pd

@pytest.fixture
def heteroatom_extractor():
    """Create OptimizedHeteroatomExtractor instance for testing"""
    return OptimizedHeteroatomExtractor()

def test_extract_heteroatoms(heteroatom_extractor):
    # Test with a small list of UniProt IDs
    uniprot_ids = ["P11511"]  # Using a real UniProt ID from our sample data
    result = heteroatom_extractor.extract_heteroatoms(uniprot_ids)
    
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert 'UniProt_ID' in result.columns
        assert 'PDB_ID' in result.columns
        assert 'Heteroatom_Code' in result.columns

def test_get_pdbs_for_uniprot(heteroatom_extractor):
    # Test PDB retrieval for a known UniProt ID
    pdbs = heteroatom_extractor.get_pdbs_for_uniprot("P11511")
    assert isinstance(pdbs, list)

def test_download_pdb(heteroatom_extractor):
    # Test PDB file download
    lines = heteroatom_extractor.download_pdb("3EQM")  # Using a known PDB ID from our data
    assert isinstance(lines, list)
    if lines:
        assert any(line.startswith("HETATM") for line in lines)

def test_fetch_smiles_rcsb(heteroatom_extractor):
    # Test SMILES fetching for a known heteroatom
    result = heteroatom_extractor.fetch_smiles_rcsb("ASD")  # Using a known heteroatom code
    assert isinstance(result, dict)
    assert 'smiles' in result
    assert 'status' in result