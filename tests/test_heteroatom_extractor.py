import pytest
from backend.heteroatom_extractor import HeteroatomExtractor
import pandas as pd

@pytest.fixture
def extractor():
    return HeteroatomExtractor()

def test_extract_heteroatoms(extractor):
    # Test with a small list of UniProt IDs
    uniprot_ids = ["P11511"]  # Using a real UniProt ID from our sample data
    result = extractor.extract_heteroatoms(uniprot_ids)
    
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert 'UniProt_ID' in result.columns
        assert 'PDB_ID' in result.columns
        assert 'Heteroatom_Code' in result.columns

def test_get_pdbs_for_uniprot(extractor):
    # Test PDB retrieval for a known UniProt ID
    pdbs = extractor.get_pdbs_for_uniprot("P11511")
    assert isinstance(pdbs, list)

def test_download_pdb(extractor):
    # Test PDB file download
    lines = extractor.download_pdb("3EQM")  # Using a known PDB ID from our data
    assert isinstance(lines, list)
    if lines:
        assert any(line.startswith("HETATM") for line in lines)

def test_fetch_smiles_rcsb(extractor):
    # Test SMILES fetching for a known heteroatom
    result = extractor.fetch_smiles_rcsb("ASD")  # Using a known heteroatom code
    assert isinstance(result, dict)
    assert 'smiles' in result
    assert 'status' in result