"""
TrackMyPDB Heteroatom Extractor
@author Anu Gamage

This module extracts heteroatoms from PDB structures associated with UniProt proteins.
Licensed under MIT License - Open Source Project
"""

import requests
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from dataclasses import dataclass
import streamlit as st

@dataclass
class HeteroatomData:
    """Data class for heteroatom information"""
    code: str
    smiles: str
    chemical_name: str
    formula: str
    status: str
    chains: Set[str]
    residue_numbers: Set[str]
    atom_names: Set[str]
    
    def to_dict(self) -> Dict:
        return {
            'Heteroatom_Code': self.code,
            'SMILES': self.smiles,
            'Chemical_Name': self.chemical_name,
            'Formula': self.formula,
            'Status': self.status,
            'Chains': ', '.join(sorted(self.chains)),
            'Residue_Numbers': ', '.join(sorted(self.residue_numbers)),
            'Atom_Count': len(self.atom_names)
        }

class OptimizedHeteroatomExtractor:
    """
    Optimized real-time heteroatom extractor for Manual Mode
    Integrates with Streamlit interface and provides efficient PDB processing
    """
    
    def __init__(self):
        self.pdbe_best_url = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"
        self.rcsb_download_url = "https://files.rcsb.org/download"
        self.rcsb_chemcomp_url = "https://data.rcsb.org/rest/v1/core/chemcomp"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TrackMyPDB/1.0 (Molecular Analysis Tool)'
        })
        
        # Cache for optimization
        self.pdb_cache = {}
        self.smiles_cache = {}
        
    def extract_heteroatoms_batch(self, uniprot_ids: List[str], 
                                progress_callback=None, 
                                max_workers: int = 5) -> pd.DataFrame:
        """
        Extract heteroatoms from multiple UniProt IDs with optimized batch processing
        Designed for Streamlit integration with progress tracking
        """
        all_records = []
        failed_pdbs = []
        total_heteroatoms = 0
        
        # Progress tracking for Streamlit
        if progress_callback is None and 'streamlit' in str(type(st)):
            progress_bar = st.progress(0)
            status_text = st.empty()
        else:
            progress_bar = None
            status_text = None
        
        total_steps = len(uniprot_ids)
        current_step = 0
        
        for uniprot_id in uniprot_ids:
            try:
                if status_text:
                    status_text.text(f"🔍 Processing {uniprot_id}...")
                
                # Get PDB structures for UniProt ID
                pdb_ids = self._get_pdbs_for_uniprot(uniprot_id)
                
                if not pdb_ids:
                    all_records.append({
                        "UniProt_ID": uniprot_id,
                        "PDB_ID": "NO_STRUCTURES",
                        "Heteroatom_Code": "NO_STRUCTURES",
                        "SMILES": "",
                        "Chemical_Name": "",
                        "Formula": "",
                        "Status": "no_structures_found",
                        "Chains": "",
                        "Residue_Numbers": "",
                        "Atom_Count": 0
                    })
                    continue
                
                # Process PDB structures with threading
                uniprot_records = self._process_pdbs_parallel(
                    uniprot_id, pdb_ids, max_workers=max_workers
                )
                
                all_records.extend(uniprot_records)
                heteroatom_count = len([r for r in uniprot_records 
                                      if r['Heteroatom_Code'] not in ['NO_HETEROATOMS', 'NO_STRUCTURES']])
                total_heteroatoms += heteroatom_count
                
                current_step += 1
                if progress_bar:
                    progress_bar.progress(current_step / total_steps)
                
            except Exception as e:
                logging.error(f"Error processing UniProt {uniprot_id}: {str(e)}")
                all_records.append({
                    "UniProt_ID": uniprot_id,
                    "PDB_ID": "ERROR",
                    "Heteroatom_Code": "ERROR",
                    "SMILES": "",
                    "Chemical_Name": "",
                    "Formula": "",
                    "Status": f"error_{str(e)[:20]}",
                    "Chains": "",
                    "Residue_Numbers": "",
                    "Atom_Count": 0
                })
        
        # Clean up progress indicators
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        
        return pd.DataFrame(all_records)
    
    def _get_pdbs_for_uniprot(self, uniprot_id: str, max_retries: int = 3) -> List[str]:
        """Get PDB IDs for given UniProt ID with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.pdbe_best_url}/{uniprot_id}", 
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                structures = []
                
                if isinstance(data, dict) and uniprot_id in data:
                    val = data[uniprot_id]
                    if isinstance(val, dict):
                        structures = val.get("best_structures", [])
                    elif isinstance(val, list):
                        structures = val
                elif isinstance(data, list):
                    structures = data
                
                pdb_ids = sorted({s["pdb_id"].upper() for s in structures if s.get("pdb_id")})
                return pdb_ids[:10]  # Limit to top 10 for efficiency
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logging.warning(f"Failed to fetch PDBs for {uniprot_id}: {str(e)}")
                return []
        
        return []
    
    def _process_pdbs_parallel(self, uniprot_id: str, pdb_ids: List[str], 
                              max_workers: int = 5) -> List[Dict]:
        """Process multiple PDB files in parallel"""
        all_records = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDB processing tasks
            future_to_pdb = {
                executor.submit(self._process_single_pdb, uniprot_id, pdb_id): pdb_id 
                for pdb_id in pdb_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pdb):
                pdb_id = future_to_pdb[future]
                try:
                    pdb_records = future.result(timeout=30)
                    all_records.extend(pdb_records)
                except Exception as e:
                    logging.error(f"Error processing PDB {pdb_id}: {str(e)}")
                    all_records.append({
                        "UniProt_ID": uniprot_id,
                        "PDB_ID": pdb_id,
                        "Heteroatom_Code": "ERROR",
                        "SMILES": "",
                        "Chemical_Name": "",
                        "Formula": "",
                        "Status": f"processing_error",
                        "Chains": "",
                        "Residue_Numbers": "",
                        "Atom_Count": 0
                    })
        
        return all_records
    
    def _process_single_pdb(self, uniprot_id: str, pdb_id: str) -> List[Dict]:
        """Process a single PDB file and extract heteroatoms"""
        try:
            # Download PDB file
            pdb_lines = self._download_pdb(pdb_id)
            if not pdb_lines:
                return [{
                    "UniProt_ID": uniprot_id,
                    "PDB_ID": pdb_id,
                    "Heteroatom_Code": "DOWNLOAD_FAILED",
                    "SMILES": "",
                    "Chemical_Name": "",
                    "Formula": "",
                    "Status": "download_failed",
                    "Chains": "",
                    "Residue_Numbers": "",
                    "Atom_Count": 0
                }]
            
            # Extract heteroatoms
            heteroatom_data = self._extract_heteroatoms_from_lines(pdb_lines)
            
            if not heteroatom_data:
                return [{
                    "UniProt_ID": uniprot_id,
                    "PDB_ID": pdb_id,
                    "Heteroatom_Code": "NO_HETEROATOMS",
                    "SMILES": "",
                    "Chemical_Name": "",
                    "Formula": "",
                    "Status": "no_heteroatoms",
                    "Chains": "",
                    "Residue_Numbers": "",
                    "Atom_Count": 0
                }]
            
            # Process each heteroatom
            records = []
            for het_code, het_info in heteroatom_data.items():
                # Get SMILES and chemical info
                smiles_data = self._get_smiles_for_heteroatom(het_code)
                
                record = {
                    "UniProt_ID": uniprot_id,
                    "PDB_ID": pdb_id,
                    **het_info.to_dict(),
                    **smiles_data
                }
                records.append(record)
                
                # Small delay to be respectful to APIs
                time.sleep(0.1)
            
            return records
            
        except Exception as e:
            logging.error(f"Error processing PDB {pdb_id}: {str(e)}")
            return [{
                "UniProt_ID": uniprot_id,
                "PDB_ID": pdb_id,
                "Heteroatom_Code": "ERROR",
                "SMILES": "",
                "Chemical_Name": "",
                "Formula": "",
                "Status": f"error_{str(e)[:20]}",
                "Chains": "",
                "Residue_Numbers": "",
                "Atom_Count": 0
            }]
    
    def _download_pdb(self, pdb_id: str) -> List[str]:
        """Download PDB file with caching"""
        if pdb_id in self.pdb_cache:
            return self.pdb_cache[pdb_id]
        
        try:
            url = f"{self.rcsb_download_url}/{pdb_id}.pdb"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            lines = response.text.splitlines()
            self.pdb_cache[pdb_id] = lines  # Cache for reuse
            return lines
            
        except Exception as e:
            logging.error(f"Error downloading {pdb_id}: {str(e)}")
            return []
    
    def _extract_heteroatoms_from_lines(self, lines: List[str]) -> Dict[str, HeteroatomData]:
        """Extract all heteroatoms from PDB lines"""
        heteroatoms = {}
        
        for line in lines:
            if not line.startswith("HETATM"):
                continue
                
            try:
                # Extract heteroatom code (columns 18-20, 1-indexed)
                het_code = line[17:20].strip()
                if not het_code:
                    continue
                
                # Get additional info
                chain = line[21:22].strip()
                res_num = line[22:26].strip()
                atom_name = line[12:16].strip()
                
                if het_code not in heteroatoms:
                    heteroatoms[het_code] = HeteroatomData(
                        code=het_code,
                        smiles="",  # Will be filled later
                        chemical_name="",
                        formula="",
                        status="",
                        chains=set(),
                        residue_numbers=set(),
                        atom_names=set()
                    )
                
                # Add details
                if chain:
                    heteroatoms[het_code].chains.add(chain)
                if res_num:
                    heteroatoms[het_code].residue_numbers.add(res_num)
                if atom_name:
                    heteroatoms[het_code].atom_names.add(atom_name)
                    
            except (IndexError, ValueError):
                continue
        
        return heteroatoms
    
    def _get_smiles_for_heteroatom(self, het_code: str) -> Dict[str, str]:
        """Get SMILES and chemical info for heteroatom with caching"""
        if het_code in self.smiles_cache:
            return self.smiles_cache[het_code]
        
        # Try RCSB first
        rcsb_data = self._fetch_smiles_rcsb(het_code)
        
        # If no SMILES from RCSB, try PubChem
        if not rcsb_data.get('SMILES'):
            pubchem_smiles = self._fetch_smiles_pubchem(het_code)
            if pubchem_smiles:
                rcsb_data['SMILES'] = pubchem_smiles
                rcsb_data['Status'] = f"{rcsb_data.get('Status', 'unknown')}_pubchem_found"
        
        # Cache the result
        self.smiles_cache[het_code] = rcsb_data
        return rcsb_data
    
    def _fetch_smiles_rcsb(self, het_code: str, max_retries: int = 2) -> Dict[str, str]:
        """Fetch SMILES from RCSB with retry logic"""
        for attempt in range(max_retries):
            try:
                url = f"{self.rcsb_chemcomp_url}/{het_code}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    smiles = data.get("rcsb_chem_comp_descriptor", {}).get("smiles", "")
                    chemical_name = data.get("chem_comp", {}).get("name", "")
                    formula = data.get("chem_comp", {}).get("formula", "")
                    
                    return {
                        'SMILES': smiles,
                        'Chemical_Name': chemical_name,
                        'Formula': formula,
                        'Status': 'success' if smiles else 'no_smiles_rcsb'
                    }
                elif response.status_code == 404:
                    return {
                        'SMILES': '',
                        'Chemical_Name': '',
                        'Formula': '',
                        'Status': 'not_in_rcsb'
                    }
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return {
                        'SMILES': '',
                        'Chemical_Name': '',
                        'Formula': '',
                        'Status': f'http_error_{response.status_code}'
                    }
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {
                    'SMILES': '',
                    'Chemical_Name': '',
                    'Formula': '',
                    'Status': f'error_{str(e)[:10]}'
                }
        
        return {
            'SMILES': '',
            'Chemical_Name': '',
            'Formula': '',
            'Status': 'failed_all_retries'
        }
    
    def _fetch_smiles_pubchem(self, het_code: str) -> str:
        """Fetch SMILES from PubChem as backup"""
        try:
            url = f"{self.pubchem_url}/name/{het_code}/property/CanonicalSMILES/JSON"
            response = self.session.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and len(props) > 0:
                    return props[0].get("CanonicalSMILES", "")
        except Exception:
            pass
        
        return ""
    
    def extract_heteroatoms_realtime_optimized(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """
        Optimized entry point for real-time heteroatom extraction
        Used by the Manual Mode interface
        """
        if not uniprot_ids:
            return pd.DataFrame()
        
        # Show progress in Streamlit
        with st.spinner(f"🔍 Extracting heteroatoms from {len(uniprot_ids)} UniProt IDs..."):
            results_df = self.extract_heteroatoms_batch(
                uniprot_ids, 
                max_workers=3  # Conservative for stability
            )
        
        # Display summary
        if not results_df.empty:
            valid_results = results_df[
                ~results_df['Heteroatom_Code'].isin(['NO_HETEROATOMS', 'NO_STRUCTURES', 'ERROR'])
            ]
            
            if not valid_results.empty:
                st.success(f"✅ Successfully extracted {len(valid_results)} heteroatom records!")
                
                # Show quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🧪 Unique Heteroatoms", valid_results['Heteroatom_Code'].nunique())
                with col2:
                    st.metric("✅ With SMILES", len(valid_results[valid_results['SMILES'] != '']))
                with col3:
                    st.metric("🏗️ PDB Structures", valid_results['PDB_ID'].nunique())
            else:
                st.warning("⚠️ No valid heteroatoms found in the specified structures")
        
        return results_df