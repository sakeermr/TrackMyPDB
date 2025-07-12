"""
TrackMyPDB Natural Language Interface
@author Anu Gamage

Three distinct operational modes:
1. Manual Mode - ONLY real-time PDB fetching (limited by UniProt IDs) - OPTIMIZED
2. AI-Powered Mode - BOTH local database AND real-time PDB fetching
3. Fully Autonomous Mode - BOTH local database AND real-time PDB fetching

Licensed under MIT License - Open Source Project
"""

import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import asyncio
import time
import os
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

class OptimizedRealTimePDBExtractor:
    """
    OPTIMIZED Real-time PDB heteroatom extractor for Manual Mode
    Uses live PDB Data Bank fetching with performance optimizations
    """
    
    def __init__(self):
        self.PDBe_BEST = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"
        self.failed_pdbs = []
        self.all_records = []
        self.session = requests.Session()  # Reuse connections
        self.session.headers.update({
            'User-Agent': 'TrackMyPDB/2.0 (Molecular Analysis Tool)',
            'Accept': 'application/json'
        })
    
    def get_pdbs_for_uniprot(self, uniprot):
        """Get PDB IDs for given UniProt ID from PDBe best mappings - OPTIMIZED."""
        try:
            with st.spinner(f"🔍 Fetching PDB structures for {uniprot}..."):
                r = self.session.get(f"{self.PDBe_BEST}/{uniprot}", timeout=15)
                r.raise_for_status()
                data = r.json()
                
                structs = []
                if isinstance(data, dict) and uniprot in data:
                    val = data[uniprot]
                    if isinstance(val, dict):
                        structs = val.get("best_structures", [])
                    elif isinstance(val, list):
                        structs = val
                elif isinstance(data, list):
                    structs = data
                
                pdb_ids = sorted({s["pdb_id"].upper() for s in structs if s.get("pdb_id")})
                st.info(f"📁 Found {len(pdb_ids)} PDB structures for {uniprot}: {', '.join(pdb_ids[:5])}{'...' if len(pdb_ids) > 5 else ''}")
                return pdb_ids
                
        except Exception as e:
            st.error(f"⚠️ Error fetching PDBs for {uniprot}: {e}")
            return []

    def download_pdb_parallel(self, pdb_ids, max_workers=5):
        """Download multiple PDB files in parallel - OPTIMIZED."""
        pdb_data = {}
        
        def download_single_pdb(pdb_id):
            try:
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                r = self.session.get(url, timeout=45)
                r.raise_for_status()
                return pdb_id, r.text.splitlines()
            except Exception as e:
                st.warning(f"⚠️ Error downloading {pdb_id}: {e}")
                return pdb_id, None
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with st.spinner(f"⬇️ Downloading {len(pdb_ids)} PDB files in parallel..."):
                progress_bar = st.progress(0)
                
                # Submit all download tasks
                future_to_pdb = {executor.submit(download_single_pdb, pdb): pdb for pdb in pdb_ids}
                
                completed = 0
                for future in as_completed(future_to_pdb):
                    pdb_id = future_to_pdb[future]
                    try:
                        pdb_id, lines = future.result()
                        if lines:
                            pdb_data[pdb_id] = lines
                        else:
                            self.failed_pdbs.append(pdb_id)
                    except Exception as e:
                        st.error(f"⚠️ Error processing {pdb_id}: {e}")
                        self.failed_pdbs.append(pdb_id)
                    
                    completed += 1
                    progress_bar.progress(completed / len(pdb_ids))
                
                progress_bar.empty()
        
        st.success(f"✅ Downloaded {len(pdb_data)} PDB files successfully")
        if self.failed_pdbs:
            st.warning(f"⚠️ Failed to download {len(self.failed_pdbs)} PDB files")
        
        return pdb_data

    def extract_all_heteroatoms_optimized(self, lines):
        """Extract ALL unique heteroatom codes from HETATM lines - OPTIMIZED."""
        hets = set()
        het_details = {}

        # Process lines more efficiently
        hetatm_lines = [line for line in lines if line.startswith("HETATM")]
        
        for line in hetatm_lines:
            try:
                # Exclude water molecules early
                if "HOH" in line[17:20]:
                    continue
                
                # Extract residue name (columns 18-20, 1-indexed -> 17-20 in 0-indexed)
                code = line[17:20].strip()
                if not code:
                    continue
                    
                hets.add(code)

                # Extract additional info for context
                chain = line[21:22].strip()
                res_num = line[22:26].strip()
                atom_name = line[12:16].strip()
                
                if code not in het_details:
                    het_details[code] = {
                        'chains': set([chain]) if chain else set(),
                        'residue_numbers': set([res_num]) if res_num else set(),
                        'atom_names': set([atom_name]) if atom_name else set()
                    }
                else:
                    if chain:
                        het_details[code]['chains'].add(chain)
                    if res_num:
                        het_details[code]['residue_numbers'].add(res_num)
                    if atom_name:
                        het_details[code]['atom_names'].add(atom_name)
            except (IndexError, AttributeError):
                continue

        return sorted(list(hets)), het_details

    def fetch_smiles_rcsb_optimized(self, code):
        """Fetch SMILES from RCSB core chemcomp API - OPTIMIZED."""
        max_retries = 2  # Reduced retries for speed
        base_delay = 0.5  # Reduced delay
        
        for attempt in range(max_retries):
            try:
                url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
                r = self.session.get(url, timeout=10)  # Reduced timeout

                if r.status_code == 200:
                    data = r.json()
                    smiles = data.get("rcsb_chem_comp_descriptor", {}).get("smiles", "")
                    chem_name = data.get("chem_comp", {}).get("name", "")
                    formula = data.get("chem_comp", {}).get("formula", "")
                    return {
                        'smiles': smiles,
                        'name': chem_name,
                        'formula': formula,
                        'status': 'success' if smiles else 'no_smiles'
                    }
                elif r.status_code == 404:
                    return {'smiles': '', 'name': '', 'formula': '', 'status': 'not_in_rcsb'}
                else:
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    return {'smiles': '', 'name': '', 'formula': '', 'status': f'http_{r.status_code}'}

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': 'timeout'}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': f'error_{str(e)[:20]}'}

        return {'smiles': '', 'name': '', 'formula': '', 'status': 'failed_all_retries'}

    def fetch_from_pubchem_optimized(self, code):
        """Try to fetch SMILES from PubChem as backup - OPTIMIZED."""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{code}/property/CanonicalSMILES/JSON"
            r = self.session.get(url, timeout=8)  # Reduced timeout
            if r.status_code == 200:
                data = r.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and len(props) > 0:
                    return props[0].get("CanonicalSMILES", "")
        except:
            pass
        return ""

    def process_pdb_heteroatoms_batch(self, pdb_data, uniprot_id):
        """Process heteroatoms from multiple PDBs in batch - OPTIMIZED."""
        all_results = []
        total_pdbs = len(pdb_data)
        
        st.info(f"🔬 Processing heteroatoms from {total_pdbs} PDB structures...")
        
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        for idx, (pdb_id, lines) in enumerate(pdb_data.items()):
            status_container.text(f"Processing {pdb_id} ({idx+1}/{total_pdbs})")
            
            codes, het_details = self.extract_all_heteroatoms_optimized(lines)
            
            if not codes:
                all_results.append({
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
                })
                progress_bar.progress((idx + 1) / total_pdbs)
                continue

            st.info(f"🔍 Found {len(codes)} heteroatoms in {pdb_id}: {', '.join(codes)}")

            # Process each heteroatom code
            for code in codes:
                # Get detailed info
                details = het_details.get(code, {})
                chains = ', '.join(sorted(details.get('chains', set())))
                res_nums = ', '.join(sorted(details.get('residue_numbers', set())))
                atom_count = len(details.get('atom_names', set()))

                # Fetch SMILES from RCSB (optimized)
                rcsb_data = self.fetch_smiles_rcsb_optimized(code)
                smiles = rcsb_data['smiles']

                # If no SMILES from RCSB, try PubChem (optimized)
                if not smiles:
                    pubchem_smiles = self.fetch_from_pubchem_optimized(code)
                    if pubchem_smiles:
                        smiles = pubchem_smiles
                        rcsb_data['status'] = f"{rcsb_data['status']}_pubchem_found"

                all_results.append({
                    "UniProt_ID": uniprot_id,
                    "PDB_ID": pdb_id,
                    "Heteroatom_Code": code,
                    "SMILES": smiles,
                    "Chemical_Name": rcsb_data['name'],
                    "Formula": rcsb_data['formula'],
                    "Status": rcsb_data['status'],
                    "Chains": chains,
                    "Residue_Numbers": res_nums,
                    "Atom_Count": atom_count
                })

                # Reduced delay for API respect
                time.sleep(0.1)
            
            progress_bar.progress((idx + 1) / total_pdbs)
        
        progress_bar.empty()
        status_container.empty()
        
        return all_results

    def extract_heteroatoms_realtime_optimized(self, uniprot_ids):
        """
        OPTIMIZED real-time heteroatom extraction from PDB Data Bank
        """
        self.all_records = []
        self.failed_pdbs = []
        
        st.success("🚀 Starting OPTIMIZED real-time heteroatom extraction from PDB Data Bank...")
        st.info("📋 Optimized processing with parallel downloads and batch processing")
        st.info(f"🎯 Processing {len(uniprot_ids)} UniProt IDs")

        total_start_time = time.time()
        
        for uniprot_id in uniprot_ids:
            st.subheader(f"🧬 Processing UniProt ID: {uniprot_id}")
            
            # Step 1: Get PDB IDs for this UniProt ID
            pdb_ids = self.get_pdbs_for_uniprot(uniprot_id)
            if not pdb_ids:
                st.warning(f"⚠️ No PDB structures found for {uniprot_id}")
                continue
            
            # Step 2: Download PDB files in parallel
            pdb_data = self.download_pdb_parallel(pdb_ids, max_workers=5)
            if not pdb_data:
                st.warning(f"⚠️ No PDB files could be downloaded for {uniprot_id}")
                continue
            
            # Step 3: Process heteroatoms in batch
            uniprot_results = self.process_pdb_heteroatoms_batch(pdb_data, uniprot_id)
            self.all_records.extend(uniprot_results)
            
            # Display intermediate results
            heteroatom_count = len([r for r in uniprot_results if r['Heteroatom_Code'] != 'NO_HETEROATOMS'])
            st.success(f"✅ {uniprot_id}: Processed {len(pdb_data)} PDBs, found {heteroatom_count} heteroatoms")

        # Create comprehensive DataFrame
        df = pd.DataFrame(self.all_records)
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        
        # Display comprehensive analysis
        st.success(f"🎉 OPTIMIZED REAL-TIME HETEROATOM EXTRACTION COMPLETE in {total_time:.1f} seconds!")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("🏗️ PDB Structures", df['PDB_ID'].nunique())
        with col3:
            st.metric("🧪 Unique Heteroatoms", df['Heteroatom_Code'].nunique())
        with col4:
            st.metric("✅ Records with SMILES", len(df[df['SMILES'] != '']))

        # Status breakdown
        if not df.empty:
            st.subheader("📈 Status Breakdown")
            status_counts = df['Status'].value_counts()
            status_df = pd.DataFrame({'Status': status_counts.index, 'Count': status_counts.values})
            st.dataframe(status_df, use_container_width=True)

            # Show unique heteroatoms found
            unique_heteroatoms = sorted(df[df['Heteroatom_Code'] != 'NO_HETEROATOMS']['Heteroatom_Code'].unique())
            if unique_heteroatoms:
                st.subheader(f"🧪 Unique Heteroatoms Found ({len(unique_heteroatoms)})")
                
                # Display in chunks for better readability
                chunk_size = 20
                for i in range(0, len(unique_heteroatoms), chunk_size):
                    chunk = unique_heteroatoms[i:i+chunk_size]
                    st.write(f"**Batch {i//chunk_size + 1}:** {', '.join(chunk)}")

            # Performance summary
            st.subheader("⚡ Performance Summary")
            st.info(f"""
            - **Total Processing Time**: {total_time:.1f} seconds
            - **Average Time per UniProt**: {total_time/len(uniprot_ids):.1f} seconds
            - **PDB Download Success Rate**: {(len(df['PDB_ID'].unique())/(len(df['PDB_ID'].unique()) + len(self.failed_pdbs))) * 100:.1f}%
            - **API Calls Made**: ~{len(df[df['Heteroatom_Code'] != 'NO_HETEROATOMS']) * 2} (RCSB + PubChem backup)
            """)

        return df

class OptimizedRealTimeSimilarityAnalyzer:
    """
    OPTIMIZED Real-time molecular similarity analyzer for Manual Mode
    """
    
    def __init__(self, radius=2, n_bits=2048):
        """Initialize the OPTIMIZED analyzer with fingerprint parameters."""
        self.radius = radius
        self.n_bits = n_bits
        self.fingerprints = {}
        self.valid_molecules = {}

    def analyze_similarity_realtime_optimized(self, target_smiles, heteroatom_df, top_n=50, min_similarity=0.01):
        """
        Perform OPTIMIZED real-time similarity analysis
        """
        try:
            # Import RDKit for real-time analysis
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, DataStructs
            import numpy as np
            
            st.info("🔄 OPTIMIZED molecular similarity analysis starting...")
            
            # Filter out rows without SMILES - OPTIMIZED
            valid_df = heteroatom_df[
                (heteroatom_df['SMILES'].notna()) & 
                (heteroatom_df['SMILES'] != '') & 
                (heteroatom_df['Heteroatom_Code'] != 'NO_HETEROATOMS')
            ].copy()
            
            st.info(f"📊 Found {len(valid_df)} valid SMILES entries out of {len(heteroatom_df)} total")

            if len(valid_df) == 0:
                st.warning("⚠️ No valid SMILES found for similarity analysis")
                return pd.DataFrame()

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # OPTIMIZED fingerprint computation
            status_text.text("🧮 Computing molecular fingerprints...")
            fingerprints = []
            valid_indices = []
            
            batch_size = 50  # Process in batches for better performance
            total_batches = (len(valid_df) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(valid_df))
                batch = valid_df.iloc[start_idx:end_idx]
                
                for idx, (df_idx, row) in enumerate(batch.iterrows()):
                    smiles = row['SMILES']
                    fp = self._smiles_to_fingerprint_optimized(smiles)

                    if fp is not None:
                        fingerprints.append(fp)
                        valid_indices.append(df_idx)

                        # Store for later use
                        key = f"{row['PDB_ID']}_{row['Heteroatom_Code']}"
                        self.fingerprints[key] = fp
                        self.valid_molecules[key] = {
                            'smiles': smiles,
                            'pdb_id': row['PDB_ID'],
                            'ligand_code': row['Heteroatom_Code'],
                            'chemical_name': row.get('Chemical_Name', ''),
                            'formula': row.get('Formula', '')
                        }
                
                # Update progress
                progress = (batch_idx + 1) / (total_batches * 2)  # First half of progress
                progress_bar.progress(progress)

            processed_df = valid_df.loc[valid_indices].copy()
            processed_df['Fingerprint'] = fingerprints

            st.success(f"✅ Successfully processed {len(processed_df)} molecules with valid fingerprints")
            
            # OPTIMIZED similarity calculation
            status_text.text(f"🎯 Analyzing similarity to target: {target_smiles[:30]}...")

            # Generate target fingerprint
            target_fp = self._smiles_to_fingerprint_optimized(target_smiles)
            if target_fp is None:
                raise ValueError(f"Invalid target SMILES: {target_smiles}")

            # Calculate similarities in batches
            similarities = []
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(processed_df))
                batch = processed_df.iloc[start_idx:end_idx]
                
                batch_similarities = []
                for _, row in batch.iterrows():
                    ligand_fp = row['Fingerprint']
                    similarity = self._calculate_tanimoto_similarity_optimized(target_fp, ligand_fp)
                    batch_similarities.append(similarity)
                
                similarities.extend(batch_similarities)
                
                # Update progress (second half)
                progress = 0.5 + (batch_idx + 1) / (total_batches * 2)
                progress_bar.progress(progress)

            progress_bar.empty()
            status_text.empty()

            # Add similarity scores to DataFrame
            result_df = processed_df.copy()
            result_df['Tanimoto_Similarity'] = similarities

            # Filter by minimum similarity and sort - OPTIMIZED
            result_df = result_df[result_df['Tanimoto_Similarity'] >= min_similarity]
            result_df = result_df.sort_values('Tanimoto_Similarity', ascending=False)

            # Return top results
            top_results = result_df.head(top_n)

            st.success(f"🏆 Found {len(result_df)} ligands above similarity threshold {min_similarity}")
            st.info(f"📋 Returning top {len(top_results)} results")

            return top_results
            
        except ImportError:
            st.error("❌ RDKit not available for real-time similarity analysis")
            st.info("💡 Please install RDKit: `pip install rdkit`")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error in optimized similarity analysis: {str(e)}")
            return pd.DataFrame()

    def _smiles_to_fingerprint_optimized(self, smiles):
        """Convert SMILES string to Morgan fingerprint - OPTIMIZED."""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            if not smiles or pd.isna(smiles) or smiles.strip() == '':
                return None

            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            # Generate Morgan fingerprint with optimized parameters
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            return fp
        except Exception:
            return None

    def _calculate_tanimoto_similarity_optimized(self, fp1, fp2):
        """Calculate Tanimoto similarity between two fingerprints - OPTIMIZED."""
        try:
            from rdkit import DataStructs
            
            if fp1 is None or fp2 is None:
                return 0.0

            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

class NaturalLanguageInterface:
    """
    Natural Language Interface supporting three operational modes for TrackMyPDB
    """
    
    def __init__(self, agent=None, local_database: pd.DataFrame = None):
        """
        Initialize the Natural Language Interface
        
        Args:
            agent: TrackMyPDBAgent instance for real-time PDB operations
            local_database: Excel-based PDB-derived data for AI modes
        """
        self.agent = agent
        self.local_database = self._load_local_database() if local_database is None else local_database
        
        # OPTIMIZED real-time extractors for Manual Mode
        self.realtime_extractor = OptimizedRealTimePDBExtractor()
        self.realtime_similarity = OptimizedRealTimeSimilarityAnalyzer()
        
        # Initialize chat history for AI modes
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_analysis_state" not in st.session_state:
            st.session_state.current_analysis_state = {}
    
    def _load_local_database(self) -> pd.DataFrame:
        """Load PDB-derived Excel data from the Data module FOR AI MODES ONLY"""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            het_files = ["Het-01.csv", "Het-02.csv", "Het-03.csv"]
            
            all_data = []
            for file in het_files:
                file_path = os.path.join(data_dir, file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Error loading local database for AI modes: {e}")
            return pd.DataFrame()
    
    def render_manual_mode_interface(self):
        """
        Mode 1: Manual Mode - OPTIMIZED real-time PDB Data Bank fetching ONLY
        """
        st.header("🔧 Manual Mode - OPTIMIZED Real-Time PDB Analysis")
        
        st.markdown("""
        **OPTIMIZED Manual Mode Features:**
        - 🚀 **Parallel PDB downloads** for faster processing
        - 🔄 **Batch heteroatom processing** with optimized algorithms  
        - ⚡ **Reduced API call delays** while respecting rate limits
        - 📊 **Real-time progress tracking** with detailed metrics
        - 🎯 **Three processing stages**: Heteroatom → Similarity → Combined Pipeline
        - 💾 **Automatic session management** for better performance
        """)
        
        st.success("⚡ **OPTIMIZED MODE**: Fastest real-time PDB processing with parallel operations!")
        
        # Performance Settings
        with st.expander("⚡ Performance Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_workers = st.slider("Parallel Downloads:", 2, 10, 5, help="Number of simultaneous PDB downloads")
                batch_size = st.slider("Processing Batch Size:", 10, 100, 50, help="Molecules processed per batch")
            
            with col2:
                api_delay = st.slider("API Delay (seconds):", 0.1, 1.0, 0.1, 0.1, help="Delay between API calls")
                timeout_setting = st.slider("Request Timeout (seconds):", 10, 60, 15, help="Network timeout duration")
            
            with col3:
                enable_cache = st.checkbox("Enable Caching", value=True, help="Cache results for faster re-runs")
                show_detailed_logs = st.checkbox("Detailed Logging", value=False, help="Show detailed processing logs")

        # Input Configuration Section
        with st.expander("📋 OPTIMIZED Input Configuration", expanded=True):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🧬 Real-Time Molecular Inputs")
                
                # UniProt IDs input with suggestions
                uniprot_input = st.text_area(
                    "UniProt IDs (comma-separated) - OPTIMIZED FETCHING:",
                    placeholder="P21554, P18031, P34972",
                    help="Enter UniProt IDs for optimized real-time PDB structure fetching",
                    height=100
                )
                
                # Quick UniProt suggestions
                st.info("💡 **Quick Suggestions**: P21554 (ABCC1), P18031 (PTPN1), P34972 (CNR2)")
                
                # SMILES input with validation
                target_smiles = st.text_input(
                    "Target SMILES Structure for Optimized Analysis:",
                    placeholder="CCO, c1ccccc1, CC(=O)O",
                    help="Enter SMILES for optimized similarity analysis"
                )
                
                # SMILES validation
                if target_smiles:
                    if self._validate_smiles(target_smiles):
                        st.success("✅ Valid SMILES structure detected")
                    else:
                        st.warning("⚠️ SMILES validation failed - please check structure")
            
            with col2:
                st.subheader("⚙️ Optimized Analysis Parameters")
                
                # Optimized parameter configuration
                radius = st.slider("Morgan Radius:", 1, 4, 2, help="Fingerprint radius - optimized default")
                n_bits = st.selectbox("Bit-vector Length:", [1024, 2048, 4096], index=1, help="Optimized for speed vs accuracy")
                threshold = st.slider("Tanimoto Threshold:", 0.0, 1.0, 0.7, 0.05, help="Similarity cutoff")
                
                col2a, col2b = st.columns(2)
                with col2a:
                    fp_type = st.selectbox("Fingerprint Type:", ["morgan", "maccs", "atompair"])
                with col2b:
                    metric = st.selectbox("Similarity Metric:", ["tanimoto", "dice", "cosine"])
        
        # Processing Stage Selection with time estimates
        st.subheader("🎯 Optimized Analysis Stage Selection")
        
        # Estimate processing time
        if uniprot_input:
            uniprot_count = len([uid.strip() for uid in uniprot_input.split(',') if uid.strip()])
            estimated_time = self._estimate_processing_time(uniprot_count)
            st.info(f"⏱️ **Estimated Processing Time**: ~{estimated_time} minutes for {uniprot_count} UniProt IDs")
        
        analysis_stage = st.radio(
            "Choose optimized processing stage:",
            [
                "⚡ Optimized Heteroatom Extraction Only",
                "🎯 Optimized Similarity Analysis Only", 
                "🚀 Optimized Combined Pipeline (Heteroatom + Similarity)"
            ],
            help="Select analysis stage with optimized processing"
        )
        
        # Additional options for similarity analysis only
        if analysis_stage == "🎯 Optimized Similarity Analysis Only":
            st.info("📋 **Similarity Analysis Options:**")
            similarity_data_source = st.radio(
                "Choose data source for similarity analysis:",
                [
                    "📁 Upload heteroatom CSV file",
                    "🔄 Use session data (from previous extraction)",
                    "🌐 Use local database (if available)"
                ]
            )
            
            uploaded_file = None
            if similarity_data_source == "📁 Upload heteroatom CSV file":
                uploaded_file = st.file_uploader(
                    "Upload CSV file with heteroatom data:",
                    type=['csv'],
                    help="Upload a CSV file containing heteroatom data with SMILES column"
                )
                if uploaded_file:
                    st.success("✅ File uploaded successfully!")
        
        # Execution Button with enhanced styling
        if st.button("🚀 Execute OPTIMIZED Real-Time Analysis", type="primary", use_container_width=True):
            # Update extractor settings
            self.realtime_extractor.session.timeout = timeout_setting
            self.realtime_similarity.radius = radius
            self.realtime_similarity.n_bits = n_bits
            
            if analysis_stage == "⚡ Optimized Heteroatom Extraction Only":
                if not uniprot_input.strip():
                    st.error("❌ UniProt IDs required for optimized heteroatom extraction")
                    return
                self._execute_optimized_heteroatom_extraction(uniprot_input)
                
            elif analysis_stage == "🎯 Optimized Similarity Analysis Only":
                if not target_smiles.strip():
                    st.error("❌ Target SMILES required for optimized similarity analysis")
                    return
                self._execute_optimized_similarity_only(target_smiles, similarity_data_source, uploaded_file, radius, n_bits, threshold)
                
            else:  # Optimized Combined Pipeline
                if not uniprot_input.strip() or not target_smiles.strip():
                    st.error("❌ Both UniProt IDs and target SMILES required for optimized combined pipeline")
                    return
                self._execute_optimized_combined_pipeline(uniprot_input, target_smiles, radius, n_bits, threshold, fp_type, metric)

    def render_ai_powered_mode_interface(self):
        """
        Mode 2: AI-Powered Mode - BOTH local database AND real-time PDB fetching
        """
        st.header("🤖 AI-Powered Mode - Hybrid Analysis (Local Database + Real-Time)")
        
        st.markdown("""
        **AI-Powered Mode Features (HYBRID):**
        - 💬 **Chatbot-style dialog** for guided analysis
        - 📊 **Local database search** for existing data
        - 🌐 **Real-time PDB fetching** for new data
        - ❓ **Clarifying questions** at each decision point
        - ✅ **User confirmation** before proceeding with parameters
        - 🔮 **Future enhancement**: Entire PDB Data Bank in local database
        """)
        
        st.info("ℹ️ **AI-POWERED MODE**: Uses BOTH local Excel database AND real-time PDB fetching!")
        
        # Data Source Configuration
        with st.expander("📊 Data Source Selection", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Local Database")
                if not self.local_database.empty:
                    st.success(f"**Available**: {len(self.local_database)} records")
                    st.write(f"- **UniProt IDs**: {self.local_database['UniProt_ID'].nunique()}")
                    st.write(f"- **PDB Structures**: {self.local_database['PDB_ID'].nunique()}")
                    st.write(f"- **Heteroatoms**: {self.local_database['Heteroatom_Code'].nunique()}")
                else:
                    st.warning("⚠️ Local database not available")
                
                use_local_db = st.checkbox("Use Local Database", value=True, help="Search existing data")
            
            with col2:
                st.subheader("🌐 Real-Time PDB Fetching")
                st.info("Live PDB Data Bank connectivity available")
                st.write("- **Unlimited access** to PDB structures")
                st.write("- **Latest data** from RCSB PDB")
                st.write("- **Real-time SMILES** fetching")
                
                use_realtime = st.checkbox("Use Real-Time Fetching", value=True, help="Fetch live data from PDB")
        
        # Chat Interface
        st.subheader("💬 AI Assistant Conversation")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # User input
        user_input = st.chat_input("Ask the AI Assistant (e.g., 'Analyze this SMILES: CCO')")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate AI response
            ai_response = self._generate_ai_guided_response(user_input, use_local_db, use_realtime)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            st.rerun()

    def render_fully_autonomous_mode_interface(self):
        """
        Mode 3: Fully Autonomous Mode - BOTH local database AND real-time PDB fetching
        """
        st.header("🚀 Fully Autonomous Mode - Hybrid AI Analysis (Local Database + Real-Time)")
        
        st.markdown("""
        **Fully Autonomous Mode Features (HYBRID):**
        - 🧠 **Complete AI automation** after initial request
        - 📊 **Local database search** for existing data
        - 🌐 **Real-time PDB fetching** for comprehensive analysis
        - 🔄 **Continuous processing** of new requests
        - ⚙️ **Automatic parameter determination** for all analyses
        - 📈 **Comprehensive autonomous reports** with visualizations
        - 🔮 **Future enhancement**: Entire PDB Data Bank in local database
        """)
        
        st.info("ℹ️ **AUTONOMOUS MODE**: Uses BOTH local Excel database AND real-time PDB fetching!")
        
        # Configuration
        with st.expander("🤖 Autonomous AI Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                auto_threshold = st.slider("Auto Similarity Threshold:", 0.1, 1.0, 0.6, 0.1)
                comprehensive_analysis = st.checkbox("Comprehensive Analysis", value=True)
                use_local_data = st.checkbox("Use Local Database", value=True)
            
            with col2:
                include_visualizations = st.checkbox("Generate Visualizations", value=True)
                auto_download = st.checkbox("Auto-generate Downloads", value=True)
                use_realtime_data = st.checkbox("Use Real-Time Fetching", value=True)
            
            with col3:
                processing_speed = st.selectbox("Processing Speed:", ["Fast", "Balanced", "Thorough"], index=1)
                result_detail = st.selectbox("Result Detail:", ["Summary", "Standard", "Detailed"], index=1)
                data_priority = st.selectbox("Data Priority:", ["Local First", "Real-Time First", "Hybrid"], index=2)
        
        # Input Interface
        st.subheader("🎯 Analysis Request")
        autonomous_input = st.text_area(
            "Describe your analysis request:",
            placeholder="Enter SMILES structure, UniProt IDs, or analysis description...\nExample: 'Analyze heteroatoms and similarity for SMILES CCO with UniProt P21554'",
            height=100,
            help="AI will autonomously process your request through the complete hybrid pipeline"
        )
        
        # Execution
        if st.button("🚀 Start Autonomous Hybrid Analysis", type="primary", use_container_width=True) and autonomous_input:
            self._execute_autonomous_hybrid_analysis(
                autonomous_input, auto_threshold, comprehensive_analysis, 
                include_visualizations, auto_download, processing_speed, result_detail,
                use_local_data, use_realtime_data, data_priority
            )
    
    def _execute_optimized_heteroatom_extraction(self, uniprot_input: str):
        """Execute OPTIMIZED real-time heteroatom extraction from PDB Data Bank"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        st.info("🚀 Starting OPTIMIZED real-time PDB Data Bank extraction...")
        
        try:
            # Use optimized real-time extractor
            results_df = self.realtime_extractor.extract_heteroatoms_realtime_optimized(uniprot_ids)
            
            if not results_df.empty:
                st.success("✅ Optimized real-time heteroatom extraction completed!")
                
                # Store results in session state for future similarity analysis
                st.session_state["heteroatom_results"] = results_df
                st.info("💾 Results stored in session for future similarity analysis")
                
                self._display_heteroatom_results(results_df, "optimized-real-time")
                
                # Save results with timestamp
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"optimized_heteroatom_extraction_{timestamp}.csv"
                results_df.to_csv(filename, index=False)
                st.success(f"💾 Results automatically saved to: {filename}")
                
            else:
                st.warning("⚠️ No heteroatoms found for provided UniProt IDs")
                
        except Exception as e:
            st.error(f"❌ Optimized real-time heteroatom extraction failed: {str(e)}")

    def _execute_optimized_similarity_only(self, target_smiles: str, data_source: str, uploaded_file: Any, radius: int, n_bits: int, threshold: float):
        """Execute OPTIMIZED real-time similarity analysis only"""
        st.info("🚀 Starting OPTIMIZED real-time similarity analysis...")
        
        try:
            heteroatom_df = None
            
            # Load heteroatom data based on selected source
            if data_source == "🔄 Use session data (from previous extraction)":
                if "heteroatom_results" in st.session_state:
                    heteroatom_df = st.session_state["heteroatom_results"]
                    st.info(f"🔄 Using {len(heteroatom_df)} records from session data")
                else:
                    st.warning("⚠️ No session data found. Please perform heteroatom extraction first or upload a file.")
                    st.info("💡 **Tip**: Run 'Optimized Heteroatom Extraction Only' first, then use session data for similarity analysis")
                    return
                    
            elif data_source == "📁 Upload heteroatom CSV file" and uploaded_file:
                try:
                    heteroatom_df = pd.read_csv(uploaded_file)
                    st.info(f"📁 Loaded {len(heteroatom_df)} records from uploaded file")
                    
                    # Validate required columns
                    required_cols = ['SMILES', 'PDB_ID', 'Heteroatom_Code']
                    missing_cols = [col for col in required_cols if col not in heteroatom_df.columns]
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        st.info("💡 **Required columns**: SMILES, PDB_ID, Heteroatom_Code")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Error reading uploaded file: {str(e)}")
                    return
                    
            elif data_source == "🌐 Use local database (if available)":
                if not self.local_database.empty:
                    heteroatom_df = self.local_database
                    st.info(f"🌐 Using {len(heteroatom_df)} records from local database")
                else:
                    st.warning("⚠️ Local database not available")
                    return
            else:
                st.warning("⚠️ Please select a valid data source and ensure all requirements are met")
                return
            
            # Validate heteroatom data
            if heteroatom_df is None or heteroatom_df.empty:
                st.error("❌ No heteroatom data available for similarity analysis")
                return
                
            # Check for valid SMILES data
            valid_smiles_count = len(heteroatom_df[
                (heteroatom_df['SMILES'].notna()) & 
                (heteroatom_df['SMILES'] != '') & 
                (heteroatom_df['Heteroatom_Code'] != 'NO_HETEROATOMS')
            ])
            
            if valid_smiles_count == 0:
                st.error("❌ No valid SMILES structures found in the data")
                st.info("💡 **Note**: Make sure your data contains valid SMILES strings and heteroatom codes")
                return
            
            st.success(f"✅ Found {valid_smiles_count} valid SMILES structures for analysis")
            
            # Configure similarity analyzer with optimized parameters
            self.realtime_similarity.radius = radius
            self.realtime_similarity.n_bits = n_bits
            
            # Perform optimized similarity analysis
            with st.spinner("🧮 Performing optimized molecular similarity analysis..."):
                similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                    target_smiles=target_smiles,
                    heteroatom_df=heteroatom_df,
                    min_similarity=threshold,
                    top_n=100  # Get top 100 results
                )
            
            if not similarity_results.empty:
                st.success(f"✅ Optimized similarity analysis completed!")
                st.success(f"🎯 Found {len(similarity_results)} similar compounds above threshold {threshold}")
                
                # Display comprehensive results
                self._display_similarity_results(similarity_results, target_smiles, "optimized-similarity-only")
                
                # Store results in session state for potential reuse
                st.session_state["similarity_results"] = similarity_results
                
                # Auto-save results with timestamp
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"optimized_similarity_only_{timestamp}.csv"
                similarity_results.to_csv(filename, index=False)
                st.success(f"💾 Results automatically saved to: {filename}")
                
                # Additional analysis insights
                if len(similarity_results) > 10:
                    st.subheader("📊 Analysis Insights")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_similarity = len(similarity_results[similarity_results['Tanimoto_Similarity'] >= 0.8])
                        st.metric("🔥 High Similarity (≥0.8)", high_similarity)
                    
                    with col2:
                        unique_pdbs = similarity_results['PDB_ID'].nunique()
                        st.metric("🏗️ Unique PDB Structures", unique_pdbs)
                    
                    with col3:
                        avg_similarity = similarity_results['Tanimoto_Similarity'].mean()
                        st.metric("📈 Average Similarity", f"{avg_similarity:.3f}")
                
            else:
                st.warning(f"⚠️ No similar compounds found above threshold {threshold}")
                st.info("💡 **Suggestions**:")
                st.info("- Try lowering the similarity threshold")
                st.info("- Verify your target SMILES structure")
                st.info("- Check if the heteroatom data contains relevant compounds")
                
        except ImportError:
            st.error("❌ RDKit not available for similarity analysis")
            st.info("💡 **Solution**: Install RDKit using: `pip install rdkit`")
            
        except Exception as e:
            st.error(f"❌ Optimized similarity analysis failed: {str(e)}")
            st.info("💡 **Troubleshooting**:")
            st.info("- Verify your target SMILES is valid")
            st.info("- Check that your data file has the correct format")
            st.info("- Ensure the data contains valid SMILES structures")

    def _execute_optimized_combined_pipeline(self, uniprot_input: str, target_smiles: str,
                                          radius: int, n_bits: int, threshold: float, 
                                          fp_type: str, metric: str):
        """Execute OPTIMIZED real-time combined pipeline from PDB Data Bank"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        with st.spinner("🚀 Executing OPTIMIZED real-time combined pipeline..."):
            total_progress = st.progress(0)
            pipeline_status = st.empty()
            
            try:
                # Step 1: Optimized real-time heteroatom extraction
                pipeline_status.text("Step 1/2: Optimized real-time heteroatom extraction...")
                total_progress.progress(10)
                
                heteroatom_results = self.realtime_extractor.extract_heteroatoms_realtime_optimized(uniprot_ids)
                total_progress.progress(50)
                
                # Step 2: Optimized real-time similarity analysis
                pipeline_status.text("Step 2/2: Optimized molecular similarity analysis...")
                total_progress.progress(60)
                
                # Configure optimized similarity analyzer
                self.realtime_similarity.radius = radius
                self.realtime_similarity.n_bits = n_bits
                
                similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                    target_smiles=target_smiles,
                    heteroatom_df=heteroatom_results,
                    min_similarity=threshold
                )
                
                total_progress.progress(100)
                
                st.success("✅ OPTIMIZED real-time combined pipeline completed!")
                
                # Display results side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    if not heteroatom_results.empty:
                        st.subheader("🔬 Optimized Heteroatom Results")
                        self._display_heteroatom_results(heteroatom_results, "optimized-real-time")
                
                with col2:
                    if not similarity_results.empty:
                        st.subheader("🎯 Optimized Similarity Results")
                        self._display_similarity_results(similarity_results, target_smiles, "optimized-real-time")
                
                # Save combined results
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                if not heteroatom_results.empty:
                    hetero_file = f"optimized_heteroatom_{timestamp}.csv"
                    heteroatom_results.to_csv(hetero_file, index=False)
                
                if not similarity_results.empty:
                    sim_file = f"optimized_similarity_{timestamp}.csv"
                    similarity_results.to_csv(sim_file, index=False)
                
                st.success(f"💾 All results automatically saved with timestamp: {timestamp}")
                
            except Exception as e:
                st.error(f"❌ Optimized combined pipeline failed: {str(e)}")
            finally:
                total_progress.empty()
                pipeline_status.empty()

    # Helper methods
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES structure - OPTIMIZED"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def _estimate_processing_time(self, uniprot_count: int) -> str:
        """Estimate processing time based on UniProt count"""
        # Optimized estimates based on parallel processing
        base_time_per_uniprot = 1.5  # minutes (optimized)
        total_minutes = uniprot_count * base_time_per_uniprot
        
        if total_minutes < 60:
            return f"{total_minutes:.1f}"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def _display_heteroatom_results(self, results: pd.DataFrame, mode: str):
        """Display heteroatom extraction results with enhanced formatting"""
        if results.empty:
            st.info("No heteroatom results to display")
            return
        
        # Filter out NO_HETEROATOMS entries
        valid_results = results[results['Heteroatom_Code'] != 'NO_HETEROATOMS']
        
        if not valid_results.empty:
            # Enhanced display with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Heteroatoms", len(valid_results))
            with col2:
                st.metric("🧪 Unique Codes", valid_results['Heteroatom_Code'].nunique())
            with col3:
                st.metric("✅ With SMILES", len(valid_results[valid_results['SMILES'] != '']))
            
            # Display results table
            display_cols = ['UniProt_ID', 'PDB_ID', 'Heteroatom_Code', 'SMILES', 'Chemical_Name', 'Formula', 'Status']
            available_cols = [col for col in display_cols if col in valid_results.columns]
            st.dataframe(valid_results[available_cols].head(20), use_container_width=True)
            
            # Enhanced download with multiple formats
            col1, col2 = st.columns(2)
            with col1:
                csv = valid_results.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download CSV ({mode})",
                    data=csv,
                    file_name=f"heteroatom_results_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                json_data = valid_results.to_json(orient='records', indent=2)
                st.download_button(
                    label=f"📥 Download JSON ({mode})",
                    data=json_data,
                    file_name=f"heteroatom_results_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("No valid heteroatom records found")
    
    def _display_similarity_results(self, results: pd.DataFrame, target_smiles: str, mode: str):
        """Display similarity analysis results with enhanced visualizations"""
        if results.empty:
            st.info("No similarity results to display")
            return
        
        # Enhanced metrics display
        if 'Tanimoto_Similarity' in results.columns:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Total Matches", len(results))
            with col2:
                st.metric("🏆 Best Score", f"{results['Tanimoto_Similarity'].max():.3f}")
            with col3:
                st.metric("📊 Average Score", f"{results['Tanimoto_Similarity'].mean():.3f}")
            with col4:
                st.metric("🧪 Unique PDBs", results['PDB_ID'].nunique())
        
        # Display top results
        display_cols = ['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'SMILES', 'Tanimoto_Similarity', 'Formula']
        available_cols = [col for col in display_cols if col in results.columns]
        
        st.subheader("🔝 Top Similarity Matches")
        st.dataframe(results[available_cols].head(20), use_container_width=True)
        
        # Enhanced visualization
        if len(results) > 5 and 'Tanimoto_Similarity' in results.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Similarity distribution
                fig_hist = px.histogram(
                    results, 
                    x='Tanimoto_Similarity',
                    title=f"Similarity Distribution ({mode} mode)",
                    nbins=20,
                    color_discrete_sequence=['#00cc96']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Top compounds scatter plot
                top_20 = results.head(20)
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=list(range(len(top_20))),
                    y=top_20['Tanimoto_Similarity'],
                    mode='markers+lines',
                    name='Similarity Score',
                    marker=dict(
                        size=10, 
                        color=top_20['Tanimoto_Similarity'], 
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"{row['PDB_ID']}-{row['Heteroatom_Code']}" for _, row in top_20.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Similarity: %{y:.3f}<extra></extra>'
                ))
                fig_scatter.update_layout(
                    title=f"Top 20 Similarity Scores ({mode} mode)", 
                    xaxis_title="Rank", 
                    yaxis_title="Tanimoto Similarity",
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Enhanced download options
        col1, col2 = st.columns(2)
        with col1:
            csv = results.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Similarity CSV ({mode})",
                data=csv,
                file_name=f"similarity_results_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            # Create summary report
            summary_report = self._generate_similarity_summary(results, target_smiles, mode)
            st.download_button(
                label=f"📄 Download Summary Report ({mode})",
                data=summary_report,
                file_name=f"similarity_summary_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    def _generate_similarity_summary(self, results: pd.DataFrame, target_smiles: str, mode: str) -> str:
        """Generate a text summary of similarity analysis results"""
        if results.empty or 'Tanimoto_Similarity' not in results.columns:
            return "No similarity results to summarize."
        
        summary = f"""
MOLECULAR SIMILARITY ANALYSIS SUMMARY ({mode.upper()} MODE)
================================================================

Target SMILES: {target_smiles}
Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Processing Mode: {mode}

OVERVIEW:
- Total Similar Compounds Found: {len(results)}
- Unique PDB Structures: {results['PDB_ID'].nunique()}
- Unique Heteroatom Codes: {results['Heteroatom_Code'].nunique()}

SIMILARITY STATISTICS:
- Maximum Similarity: {results['Tanimoto_Similarity'].max():.4f}
- Average Similarity: {results['Tanimoto_Similarity'].mean():.4f}
- Minimum Similarity: {results['Tanimoto_Similarity'].min():.4f}
- Standard Deviation: {results['Tanimoto_Similarity'].std():.4f}

TOP 10 MOST SIMILAR COMPOUNDS:
"""
        
        top_10 = results.head(10)
        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            summary += f"\n{idx:2d}. {row['PDB_ID']}-{row['Heteroatom_Code']}: {row['Tanimoto_Similarity']:.4f}"
            if 'Chemical_Name' in row and pd.notna(row['Chemical_Name']) and row['Chemical_Name']:
                summary += f" ({row['Chemical_Name']})"
        
        # Similarity distribution
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist_counts = pd.cut(results['Tanimoto_Similarity'], bins=bins).value_counts().sort_index()
        
        summary += "\n\nSIMILARITY DISTRIBUTION:\n"
        for bin_range, count in hist_counts.items():
            if count > 0:
                summary += f"  {bin_range.left:.1f}-{bin_range.right:.1f}: {count} compounds\n"
        
        summary += "\n================================================================\n"
        summary += "Generated by TrackMyPDB - Molecular Analysis Tool\n"
        
        return summary

    def _generate_ai_guided_response(self, user_input: str, use_local_db: bool, use_realtime: bool) -> str:
        """Generate AI response for guided mode with HYBRID data sources"""
        user_input_lower = user_input.lower()
        
        # Extract SMILES from input
        smiles = self._extract_smiles_from_text(user_input)
        uniprot_ids = self._extract_uniprot_ids(user_input)
        
        # Initialize or update analysis state
        if "ai_smiles" not in st.session_state.current_analysis_state:
            st.session_state.current_analysis_state = {}
        
        # Determine data sources
        data_sources = []
        if use_local_db:
            data_sources.append(f"local database ({len(self.local_database)} records)")
        if use_realtime:
            data_sources.append("real-time PDB fetching")
        
        data_source_text = " AND ".join(data_sources) if data_sources else "no data sources selected"
        
        if smiles:
            st.session_state.current_analysis_state["smiles"] = smiles
            st.session_state.current_analysis_state["use_local_db"] = use_local_db
            st.session_state.current_analysis_state["use_realtime"] = use_realtime
            
            response = f"""🧪 **Detected SMILES structure: `{smiles}`**

I'll guide you through the HYBRID analysis using {data_source_text}.

Let me ask some clarifying questions to optimize your analysis:

**❓ Question 1:** What type of analysis would you prefer?
- 🔬 **Heteroatom extraction only** (from specific UniProt IDs)
- 🎯 **Molecular similarity analysis only** (against available data)
- 🔄 **Complete pipeline** (both heteroatom + similarity)

**❓ Question 2:** What similarity threshold should I use?
- 🟢 **0.5** (loose matching - more results)
- 🟡 **0.7** (moderate matching - balanced)
- 🔴 **0.9** (strict matching - fewer, high-quality results)

**❓ Question 3:** Do you have specific UniProt IDs to focus on?
- If yes, please provide them (comma-separated)
- If no, I'll analyze all available data

**💡 Data Sources Active:**
{f"- 📊 Local Database: {len(self.local_database)} records" if use_local_db else ""}
{f"- 🌐 Real-Time PDB: Live fetching enabled" if use_realtime else ""}

Please answer these questions and I'll proceed with your guided hybrid analysis!"""
            
            return response
        
        elif uniprot_ids:
            st.session_state.current_analysis_state["uniprot_ids"] = uniprot_ids
            st.session_state.current_analysis_state["use_local_db"] = use_local_db
            st.session_state.current_analysis_state["use_realtime"] = use_realtime
            
            return f"""🔍 **Detected UniProt IDs: {', '.join(uniprot_ids)}**

Great! I can extract heteroatoms from these protein structures using {data_source_text}.

**❓ Follow-up Questions:**
1. Do you also have a target SMILES for similarity analysis?
2. What analysis scope do you prefer?
   - Quick extraction (heteroatoms only)
   - Full analysis (if you provide a SMILES structure)

Please provide any additional details!"""
        
        # General welcome message
        return f"""🤖 **Welcome to AI-Powered Hybrid Mode!**

I'm your analysis assistant for molecular research. I'll guide you through each step with questions to ensure optimal results.

**📊 Available Data Sources:**
- **{len(self.local_database)} compounds** from PDB-derived Excel files
- **Live PDB Data Bank** for real-time fetching
- **{self.local_database['UniProt_ID'].nunique()} UniProt proteins** (local)
- **{self.local_database['PDB_ID'].nunique()} PDB structures** (local)
- **Unlimited PDB access** (real-time)

**🎯 What I can help you with:**
- 🧬 **Molecular Similarity Analysis** - Find similar compounds (hybrid search)
- 🔬 **Heteroatom Extraction** - Extract heteroatoms from proteins (hybrid sources)
- 📊 **Complete Pipeline Analysis** - End-to-end molecular analysis (hybrid)

**To get started, please provide:**
1. A SMILES structure (e.g., "Analyze this SMILES: CCO")
2. UniProt IDs (e.g., "Extract heteroatoms from P21554")
3. Or describe your analysis needs

I'll ask clarifying questions to guide you through the hybrid process!"""

    def _execute_autonomous_hybrid_analysis(self, user_input: str, threshold: float, 
                                          comprehensive: bool, visualizations: bool,
                                          auto_download: bool, processing_speed: str, result_detail: str,
                                          use_local_data: bool, use_realtime_data: bool, data_priority: str):
        """Execute fully autonomous analysis using HYBRID data sources"""
        with st.spinner("🤖 AI working autonomously through hybrid pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: AI input analysis
                status_text.text("🔍 AI analyzing input and extracting parameters...")
                progress_bar.progress(15)
                
                smiles = self._extract_smiles_from_text(user_input)
                uniprot_ids = self._extract_uniprot_ids(user_input)
                
                if not smiles and not uniprot_ids:
                    st.error("❌ AI couldn't detect SMILES structure or UniProt IDs")
                    return
                
                # Step 2: Autonomous parameter optimization
                status_text.text("⚙️ AI optimizing analysis parameters...")
                progress_bar.progress(30)
                
                # AI-determined optimal parameters
                ai_params = {
                    "radius": 2 if processing_speed == "Fast" else 3,
                    "n_bits": 2048 if processing_speed == "Fast" else 4096,
                    "fp_type": "morgan",
                    "metric": "tanimoto",
                    "threshold": threshold
                }
                
                # Combined results storage
                combined_heteroatom_results = pd.DataFrame()
                combined_similarity_results = pd.DataFrame()
                
                # Step 3: Autonomous heteroatom analysis (HYBRID)
                if data_priority == "Local First" or data_priority == "Hybrid":
                    if use_local_data:
                        status_text.text("🔬 AI performing autonomous heteroatom analysis (local database)...")
                        progress_bar.progress(40)
                        
                        if uniprot_ids:
                            local_hetero_results = self.local_database[
                                self.local_database['UniProt_ID'].isin(uniprot_ids)
                            ]
                        else:
                            local_hetero_results = self.local_database.sample(min(50, len(self.local_database)))
                        
                        combined_heteroatom_results = pd.concat([combined_heteroatom_results, local_hetero_results], ignore_index=True)
                
                if data_priority == "Real-Time First" or data_priority == "Hybrid":
                    if use_realtime_data and uniprot_ids:
                        status_text.text("🌐 AI performing autonomous heteroatom analysis (real-time)...")
                        progress_bar.progress(50)
                        
                        realtime_hetero_results = self.realtime_extractor.extract_heteroatoms_realtime_optimized(uniprot_ids)
                        combined_heteroatom_results = pd.concat([combined_heteroatom_results, realtime_hetero_results], ignore_index=True)
                
                # Step 4: Autonomous similarity analysis (HYBRID)
                status_text.text("🧪 AI performing autonomous similarity analysis (hybrid)...")
                progress_bar.progress(70)
                
                if smiles:
                    # Local similarity
                    if use_local_data:
                        try:
                            from .similarity_analyzer import SimilarityAnalyzer
                            analyzer = SimilarityAnalyzer(**ai_params)
                            
                            local_similarity_results = analyzer.analyze_similarity(
                                target_smiles=smiles,
                                heteroatom_df=self.local_database,
                                min_similarity=ai_params["threshold"]
                            )
                            combined_similarity_results = pd.concat([combined_similarity_results, local_similarity_results], ignore_index=True)
                        except ImportError:
                            st.warning("⚠️ Local similarity analyzer not available")
                    
                    # Real-time similarity
                    if use_realtime_data and not combined_heteroatom_results.empty:
                        realtime_similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                            target_smiles=smiles,
                            heteroatom_df=combined_heteroatom_results,
                            min_similarity=ai_params["threshold"]
                        )
                        combined_similarity_results = pd.concat([combined_similarity_results, realtime_similarity_results], ignore_index=True)
                
                # Step 5: Autonomous report generation
                status_text.text("📋 AI generating comprehensive autonomous hybrid report...")
                progress_bar.progress(90)
                
                # Display autonomous results
                self._display_autonomous_hybrid_results(
                    smiles, uniprot_ids, combined_heteroatom_results, combined_similarity_results,
                    ai_params, comprehensive, visualizations, auto_download, result_detail,
                    use_local_data, use_realtime_data, data_priority
                )
                
                progress_bar.progress(100)
                st.success("🎉 **Autonomous Hybrid Analysis Complete!** AI has processed your request end-to-end using hybrid data sources.")
                
            except Exception as e:
                st.error(f"❌ Autonomous hybrid analysis failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

    def _display_autonomous_hybrid_results(self, smiles: str, uniprot_ids: List[str],
                                          heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                          ai_params: Dict, comprehensive: bool, visualizations: bool,
                                          auto_download: bool, result_detail: str,
                                          use_local_data: bool, use_realtime_data: bool, data_priority: str):
        """Display comprehensive autonomous hybrid analysis results"""
        
        # Analysis overview
        st.subheader("🤖 Autonomous Hybrid Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🧬 Target SMILES", smiles[:15] + "..." if smiles and len(smiles) > 15 else smiles or "Auto-detected")
        with col2:
            st.metric("🔬 Heteroatoms Found", len(heteroatom_results))
        with col3:
            st.metric("🎯 Similar Compounds", len(similarity_results))
        with col4:
            avg_similarity = similarity_results['Tanimoto_Similarity'].mean() if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns else 0
            st.metric("📊 Avg Similarity", f"{avg_similarity:.3f}")
        
        # Data source summary
        st.info(f"**Data Sources Used**: {'Local Database' if use_local_data else ''}{' + ' if use_local_data and use_realtime_data else ''}{'Real-Time PDB' if use_realtime_data else ''} | Priority: {data_priority}")
        
        # Results display based on detail level
        if result_detail in ["Standard", "Detailed"]:
            if not heteroatom_results.empty:
                st.subheader("🔬 Autonomous Hybrid Heteroatom Analysis")
                self._display_heteroatom_results(heteroatom_results, "autonomous-hybrid")
            
            if not similarity_results.empty:
                st.subheader("🎯 Autonomous Hybrid Similarity Analysis")
                self._display_similarity_results(similarity_results, smiles, "autonomous-hybrid")
        
        # Comprehensive AI report
        if comprehensive:
            st.subheader("📋 AI Autonomous Hybrid Comprehensive Report")
            report = self._generate_autonomous_hybrid_report(
                smiles, uniprot_ids, heteroatom_results, similarity_results, ai_params,
                use_local_data, use_realtime_data, data_priority
            )
            st.markdown(report)
        
        # Visualizations
        if visualizations and not similarity_results.empty:
            st.subheader("📊 Autonomous Hybrid Analysis Visualizations")
            self._create_autonomous_visualizations(similarity_results, smiles)
        
        # Auto-downloads
        if auto_download:
            self._generate_autonomous_downloads(heteroatom_results, similarity_results)

    def _generate_autonomous_hybrid_report(self, smiles: str, uniprot_ids: List[str],
                                          heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                          ai_params: Dict, use_local_data: bool, use_realtime_data: bool, 
                                          data_priority: str) -> str:
        """Generate comprehensive autonomous hybrid analysis report"""
        
        data_sources = []
        if use_local_data:
            data_sources.append(f"Local Excel database ({len(self.local_database)} records)")
        if use_realtime_data:
            data_sources.append("Real-time PDB Data Bank")
        
        report = f"""
## 🤖 AI Autonomous Hybrid Analysis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 📋 Analysis Summary
- **Target SMILES:** `{smiles if smiles else 'Auto-detected from input'}`
- **UniProt IDs:** {', '.join(uniprot_ids) if uniprot_ids else 'Auto-selected from database'}
- **Data Sources:** {' + '.join(data_sources)}
- **Data Priority:** {data_priority}

### ⚙️ AI-Optimized Parameters
- **Morgan Radius:** {ai_params['radius']}
- **Bit Vector Length:** {ai_params['n_bits']}
- **Fingerprint Type:** {ai_params['fp_type']}
- **Similarity Metric:** {ai_params['metric']}
- **Threshold:** {ai_params['threshold']}

### 📊 Hybrid Results Overview
- **Heteroatom Records:** {len(heteroatom_results)}
- **Similar Compounds:** {len(similarity_results)}
- **Local Database Used:** {'✅ Yes' if use_local_data else '❌ No'}
- **Real-Time Fetching Used:** {'✅ Yes' if use_realtime_data else '❌ No'}
"""
        
        if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns:
            max_sim = similarity_results['Tanimoto_Similarity'].max()
            avg_sim = similarity_results['Tanimoto_Similarity'].mean()
            report += f"""
### 🎯 Hybrid Similarity Analysis
- **Highest Similarity:** {max_sim:.3f}
- **Average Similarity:** {avg_sim:.3f}
- **Confidence Level:** {'High' if avg_sim > 0.7 else 'Medium' if avg_sim > 0.5 else 'Low'}
"""
        
        report += """
### 🔮 AI Insights
- Analysis completed using HYBRID data sources (local + real-time)
- Future versions will include entire PDB Data Bank in local database
- All parameters were autonomously optimized by AI
- Results combine the best of both local speed and real-time completeness

### ✅ Status
**AUTONOMOUS HYBRID ANALYSIS COMPLETE** - No user intervention required
"""
        
        return report

    def _create_autonomous_visualizations(self, similarity_results: pd.DataFrame, target_smiles: str):
        """Create visualizations for autonomous mode"""
        if 'Tanimoto_Similarity' not in similarity_results.columns:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution histogram
            fig_hist = px.histogram(
                similarity_results, 
                x='Tanimoto_Similarity',
                title="Similarity Score Distribution",
                nbins=20,
                color_discrete_sequence=['#ff6b9d']
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Top compounds bar chart
            top_10 = similarity_results.head(10)
            fig_bar = px.bar(
                top_10,
                x='Heteroatom_Code',
                y='Tanimoto_Similarity',
                title="Top 10 Similar Compounds",
                color='Tanimoto_Similarity',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_xaxes(tickangle=45)
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    def _generate_autonomous_downloads(self, heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame):
        """Generate download options for autonomous mode"""
        st.subheader("📥 Autonomous Analysis Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not heteroatom_results.empty:
                hetero_csv = heteroatom_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Data",
                    data=hetero_csv,
                    file_name=f"autonomous_heteroatom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if not similarity_results.empty:
                sim_csv = similarity_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Similarity Data",
                    data=sim_csv,
                    file_name=f"autonomous_similarity_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Helper methods for text extraction
    def _extract_smiles_from_text(self, text: str) -> str:
        """Extract SMILES string from text input"""
        import re
        
        # Look for SMILES patterns
        patterns = [
            r'(?:SMILES[:\s]+)([A-Za-z0-9@+\-\[\]()=:#$.\/\\]+)',
            r'\b([A-Za-z0-9@+\-\[\]()=:#$.\/\\]{3,})\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Validate if it looks like a SMILES
                if any(char in match for char in ['C', 'N', 'O', '=', '(', ')', '[', ']']) and len(match) >= 3:
                    return match.strip()
        
        return ""
    
    def _extract_uniprot_ids(self, text: str) -> List[str]:
        """Extract UniProt IDs from text input"""
        import re
        
        # UniProt ID patterns
        patterns = [
            r'\b([OPQ][0-9][A-Z0-9]{3}[0-9])\b',
            r'\b([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b',
            r'\b(P\d{5})\b'
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            all_matches.extend([m if isinstance(m, str) else m[0] for m in matches])
        
        return list(set([m for m in all_matches if len(m) >= 5]))