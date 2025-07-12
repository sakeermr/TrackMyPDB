"""
TrackMyPDB Natural Language Interface
@author Anu Gamage

Three distinct operational modes:
1. Manual Mode - ONLY real-time PDB fetching (limited by UniProt IDs)
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
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class RealTimePDBExtractor:
    """
    Real-time PDB heteroatom extractor for all modes
    Uses live PDB Data Bank fetching
    """
    
    def __init__(self):
        self.PDBe_BEST = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures"
        self.failed_pdbs = []
        self.all_records = []
    
    def get_pdbs_for_uniprot(self, uniprot):
        """Get PDB IDs for given UniProt ID from PDBe best mappings."""
        try:
            r = requests.get(f"{self.PDBe_BEST}/{uniprot}", timeout=10)
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
            return sorted({s["pdb_id"].upper() for s in structs if s.get("pdb_id")})
        except Exception as e:
            st.error(f"⚠️ Error fetching PDBs for {uniprot}: {e}")
            return []

    def download_pdb(self, pdb):
        """Download .pdb file and return lines."""
        try:
            url = f"https://files.rcsb.org/download/{pdb}.pdb"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text.splitlines()
        except Exception as e:
            st.warning(f"⚠️ Error downloading {pdb}: {e}")
            return []

    def extract_all_heteroatoms(self, lines):
        """Extract ALL unique heteroatom codes from HETATM lines - NO FILTERING."""
        hets = set()
        het_details = {}

        for line in lines:
            if line.startswith("HETATM"):
                # Extract residue name (columns 18-20, 1-indexed -> 17-20 in 0-indexed)
                code = line[17:20].strip()
                if code:  # Only add non-empty codes
                    hets.add(code)

                    # Also extract additional info for context
                    if code not in het_details:
                        try:
                            chain = line[21:22].strip()
                            res_num = line[22:26].strip()
                            atom_name = line[12:16].strip()
                            het_details[code] = {
                                'chains': set([chain]) if chain else set(),
                                'residue_numbers': set([res_num]) if res_num else set(),
                                'atom_names': set([atom_name]) if atom_name else set()
                            }
                        except:
                            het_details[code] = {'chains': set(), 'residue_numbers': set(), 'atom_names': set()}
                    else:
                        try:
                            chain = line[21:22].strip()
                            res_num = line[22:26].strip()
                            atom_name = line[12:16].strip()
                            if chain:
                                het_details[code]['chains'].add(chain)
                            if res_num:
                                het_details[code]['residue_numbers'].add(res_num)
                            if atom_name:
                                het_details[code]['atom_names'].add(atom_name)
                        except:
                            pass

        return sorted(list(hets)), het_details

    def fetch_smiles_rcsb(self, code):
        """Fetch SMILES from RCSB core chemcomp API with comprehensive error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
                r = requests.get(url, timeout=15)

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
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {'smiles': '', 'name': '', 'formula': '', 'status': f'http_{r.status_code}'}

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': 'timeout'}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {'smiles': '', 'name': '', 'formula': '', 'status': f'error_{str(e)[:20]}'}

        return {'smiles': '', 'name': '', 'formula': '', 'status': 'failed_all_retries'}

    def fetch_from_pubchem(self, code):
        """Try to fetch SMILES from PubChem as backup."""
        try:
            # Try by compound name
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{code}/property/CanonicalSMILES/JSON"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and len(props) > 0:
                    return props[0].get("CanonicalSMILES", "")
        except:
            pass
        return ""

    def process_pdb_heteroatoms(self, pdb_id, uniprot_id, lines):
        """Process all heteroatoms from a single PDB."""
        codes, het_details = self.extract_all_heteroatoms(lines)
        results = []

        if not codes:
            results.append({
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
            return results

        st.info(f"🔍 Processing {len(codes)} heteroatoms from {pdb_id}: {', '.join(codes)}")

        for code in codes:
            # Get detailed info
            details = het_details.get(code, {})
            chains = ', '.join(sorted(details.get('chains', set())))
            res_nums = ', '.join(sorted(details.get('residue_numbers', set())))
            atom_count = len(details.get('atom_names', set()))

            # Fetch SMILES from RCSB
            rcsb_data = self.fetch_smiles_rcsb(code)
            smiles = rcsb_data['smiles']

            # If no SMILES from RCSB, try PubChem
            if not smiles:
                pubchem_smiles = self.fetch_from_pubchem(code)
                if pubchem_smiles:
                    smiles = pubchem_smiles
                    rcsb_data['status'] = f"{rcsb_data['status']}_pubchem_found"

            results.append({
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

            # Small delay to be respectful to APIs
            time.sleep(0.2)

        return results

    def extract_heteroatoms_realtime(self, uniprot_ids, progress_callback=None):
        """
        Extract heteroatoms in real-time from PDB Data Bank
        """
        self.all_records = []
        self.failed_pdbs = []
        total_heteroatoms = 0
        
        st.info("🚀 Starting REAL-TIME heteroatom extraction from PDB Data Bank...")
        st.info("📋 This fetches EVERY heteroatom from ALL PDB files in real-time")
        st.info(f"🎯 Processing {len(uniprot_ids)} UniProt IDs")
        st.warning("⏱️ This may take a while due to real-time API calls...")

        # Calculate total PDBs for progress
        total_pdbs = 0
        uniprot_pdb_mapping = {}
        
        for up in uniprot_ids:
            pdbs = self.get_pdbs_for_uniprot(up)
            uniprot_pdb_mapping[up] = pdbs
            total_pdbs += len(pdbs)
            st.info(f"📁 Found {len(pdbs)} PDB structures for {up}")

        current_progress = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        for up in uniprot_ids:
            pdbs = uniprot_pdb_mapping[up]
            
            for pdb in pdbs:
                try:
                    status_text.text(f"Processing {pdb} for {up} ({current_progress+1}/{total_pdbs})")
                    
                    # Download PDB file
                    lines = self.download_pdb(pdb)
                    if not lines:
                        self.failed_pdbs.append(pdb)
                        current_progress += 1
                        progress_bar.progress(current_progress / total_pdbs if total_pdbs > 0 else 0)
                        continue

                    # Process all heteroatoms
                    pdb_results = self.process_pdb_heteroatoms(pdb, up, lines)
                    self.all_records.extend(pdb_results)

                    # Count heteroatoms found
                    heteroatom_count = len([r for r in pdb_results if r['Heteroatom_Code'] != 'NO_HETEROATOMS'])
                    total_heteroatoms += heteroatom_count

                except Exception as e:
                    st.error(f"⚠️ Error processing {pdb}: {e}")
                    self.failed_pdbs.append(pdb)
                    
                current_progress += 1
                progress_bar.progress(current_progress / total_pdbs if total_pdbs > 0 else 0)

        progress_bar.empty()
        status_text.empty()

        # Create comprehensive DataFrame
        df = pd.DataFrame(self.all_records)
        
        # Display comprehensive analysis
        st.success("🎉 REAL-TIME HETEROATOM EXTRACTION COMPLETE!")
        st.write(f"📊 Total records: {len(df)}")
        st.write(f"🏗️ PDB structures processed: {df['PDB_ID'].nunique()}")
        st.write(f"🧪 Total unique heteroatoms found: {df['Heteroatom_Code'].nunique()}")
        st.write(f"✅ Records with SMILES: {len(df[df['SMILES'] != ''])}")
        st.write(f"⚠️ Failed PDB downloads: {len(self.failed_pdbs)}")

        # Status breakdown
        st.subheader("📈 STATUS BREAKDOWN:")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            st.write(f"   {status}: {count}")

        # Show unique heteroatoms found
        if not df.empty:
            unique_heteroatoms = sorted(df[df['Heteroatom_Code'] != 'NO_HETEROATOMS']['Heteroatom_Code'].unique())
            st.write(f"🧪 ALL UNIQUE HETEROATOMS FOUND ({len(unique_heteroatoms)}):")
            st.write(f"   {', '.join(unique_heteroatoms)}")

            # Show heteroatoms WITH SMILES
            with_smiles = df[df['SMILES'] != '']['Heteroatom_Code'].unique()
            st.write(f"✅ HETEROATOMS WITH SMILES ({len(with_smiles)}):")
            st.write(f"   {', '.join(sorted(with_smiles))}")

        return df

class RealTimeSimilarityAnalyzer:
    """
    Real-time molecular similarity analyzer for all modes
    """
    
    def __init__(self, radius=3, n_bits=4096):
        """Initialize the analyzer with fingerprint parameters."""
        self.radius = radius
        self.n_bits = n_bits
        self.fingerprints = {}
        self.valid_molecules = {}

    def analyze_similarity_realtime(self, target_smiles, heteroatom_df, top_n=50, min_similarity=0.01):
        """
        Perform real-time similarity analysis using the provided backend code
        """
        try:
            # Import RDKit for real-time analysis
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, DataStructs
            import numpy as np
            
            st.info("🔄 Processing DataFrame and computing fingerprints...")
            
            # Filter out rows without SMILES
            valid_df = heteroatom_df[heteroatom_df['SMILES'].notna() & (heteroatom_df['SMILES'] != '')].copy()
            st.info(f"📊 Found {len(valid_df)} entries with SMILES out of {len(heteroatom_df)} total")

            # Compute fingerprints
            fingerprints = []
            valid_indices = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (df_idx, row) in enumerate(valid_df.iterrows()):
                status_text.text(f"Computing fingerprints... {idx+1}/{len(valid_df)}")
                
                smiles = row['SMILES']
                fp = self._smiles_to_fingerprint(smiles)

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
                
                progress_bar.progress((idx + 1) / len(valid_df))

            progress_bar.empty()
            status_text.empty()

            processed_df = valid_df.loc[valid_indices].copy()
            processed_df['Fingerprint'] = fingerprints

            st.success(f"✅ Successfully processed {len(processed_df)} molecules with valid fingerprints")
            
            # Find similar ligands
            st.info(f"🎯 Analyzing similarity to target: {target_smiles}")

            # Generate target fingerprint
            target_fp = self._smiles_to_fingerprint(target_smiles)
            if target_fp is None:
                raise ValueError(f"Invalid target SMILES: {target_smiles}")

            # Calculate similarities
            similarities = []
            st.info("🔍 Computing similarities...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (_, row) in enumerate(processed_df.iterrows()):
                status_text.text(f"Computing similarities... {idx+1}/{len(processed_df)}")
                ligand_fp = row['Fingerprint']
                similarity = self._calculate_tanimoto_similarity(target_fp, ligand_fp)
                similarities.append(similarity)
                
                progress_bar.progress((idx + 1) / len(processed_df))

            progress_bar.empty()
            status_text.empty()

            # Add similarity scores to DataFrame
            result_df = processed_df.copy()
            result_df['Tanimoto_Similarity'] = similarities

            # Filter by minimum similarity and sort
            result_df = result_df[result_df['Tanimoto_Similarity'] >= min_similarity]
            result_df = result_df.sort_values('Tanimoto_Similarity', ascending=False)

            # Return top results
            top_results = result_df.head(top_n)

            st.success(f"🏆 Found {len(result_df)} ligands above similarity threshold {min_similarity}")
            st.info(f"📋 Returning top {len(top_results)} results")

            return top_results
            
        except ImportError:
            st.error("❌ RDKit not available for real-time similarity analysis")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error in real-time similarity analysis: {str(e)}")
            return pd.DataFrame()

    def _smiles_to_fingerprint(self, smiles):
        """Convert SMILES string to Morgan fingerprint."""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            if not smiles or pd.isna(smiles) or smiles.strip() == '':
                return None

            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None

            # Generate Morgan fingerprint
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            return fp
        except Exception as e:
            return None

    def _calculate_tanimoto_similarity(self, fp1, fp2):
        """Calculate Tanimoto similarity between two fingerprints."""
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
        
        # Real-time extractors for all modes
        self.realtime_extractor = RealTimePDBExtractor()
        self.realtime_similarity = RealTimeSimilarityAnalyzer()
        
        # Initialize chat history for AI modes
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_analysis_state" not in st.session_state:
            st.session_state.current_analysis_state = {}
    
    def _load_local_database(self) -> pd.DataFrame:
        """
        Load PDB-derived Excel data from the Data module
        FOR AI MODES ONLY - Manual Mode does NOT use this
        
        Returns:
            pd.DataFrame: Combined data from all Het-*.csv files
        """
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
        Mode 1: Manual Mode - ONLY real-time PDB Data Bank fetching (limited by UniProt IDs)
        """
        st.header("🔧 Manual Mode - REAL-TIME PDB Data Bank Analysis")
        
        st.markdown("""
        **Manual Mode Features (REAL-TIME ONLY):**
        - 🌐 **EXCLUSIVE real-time PDB Data Bank fetching** (limited by UniProt IDs)
        - 📝 **Manual parameter configuration** for Morgan fingerprints
        - ⚙️ **User-controlled analysis settings** (radius, bit-vector length, thresholds)
        - 🎯 **Three processing stages**: Heteroatom Extraction → Similarity Analysis → Combined Pipeline
        - 📊 **Live data from PDB** with immediate results display
        """)
        
        st.warning("⚠️ **MANUAL MODE**: Uses ONLY real-time PDB Data Bank - limited by UniProt IDs!")
        
        # Input Configuration Section
        with st.expander("📋 Real-Time Input Configuration", expanded=True):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🧬 Real-Time Molecular Inputs")
                
                # UniProt IDs input
                uniprot_input = st.text_area(
                    "UniProt IDs (comma-separated) - REAL-TIME FETCHING:",
                    placeholder="P21554, P18031, P34972",
                    help="Enter UniProt IDs for REAL-TIME PDB structure fetching from PDB Data Bank",
                    height=100
                )
                
                # SMILES input
                target_smiles = st.text_input(
                    "Target SMILES Structure for Real-Time Analysis:",
                    placeholder="CCO, c1ccccc1, CC(=O)O",
                    help="Enter the SMILES string for real-time similarity analysis against fetched data"
                )
            
            with col2:
                st.subheader("⚙️ Real-Time Analysis Parameters")
                
                # Manual parameter configuration
                radius = st.slider("Morgan Radius:", 1, 4, 2, help="Fingerprint radius for molecular comparison")
                n_bits = st.selectbox("Bit-vector Length:", [1024, 2048, 4096], index=1, help="Fingerprint resolution")
                threshold = st.slider("Tanimoto Threshold:", 0.0, 1.0, 0.7, 0.05, help="Minimum similarity score")
                
                col2a, col2b = st.columns(2)
                with col2a:
                    fp_type = st.selectbox("Fingerprint Type:", ["morgan", "maccs", "atompair"])
                with col2b:
                    metric = st.selectbox("Similarity Metric:", ["tanimoto", "dice", "cosine"])
        
        # Processing Stage Selection
        st.subheader("🎯 Real-Time Analysis Stage Selection")
        analysis_stage = st.radio(
            "Choose real-time processing stage:",
            [
                "Real-Time Heteroatom Extraction Only",
                "Real-Time Molecular Similarity Analysis Only", 
                "Real-Time Combined Pipeline (Heteroatom + Similarity)"
            ],
            help="Select which real-time analysis stage to execute from PDB Data Bank"
        )
        
        # Execution Button
        if st.button("🚀 Execute REAL-TIME Analysis (Live PDB Data Bank)", type="primary", use_container_width=True):
            if analysis_stage == "Real-Time Heteroatom Extraction Only":
                if not uniprot_input.strip():
                    st.error("❌ UniProt IDs required for real-time heteroatom extraction")
                    return
                self._execute_realtime_heteroatom_extraction(uniprot_input)
                
            elif analysis_stage == "Real-Time Molecular Similarity Analysis Only":
                if not target_smiles.strip():
                    st.error("❌ Target SMILES required for real-time similarity analysis")
                    return
                st.error("❌ Real-time similarity analysis requires heteroatom data first. Use Combined Pipeline instead.")
                
            else:  # Real-Time Combined Pipeline
                if not uniprot_input.strip() or not target_smiles.strip():
                    st.error("❌ Both UniProt IDs and target SMILES required for real-time combined pipeline")
                    return
                self._execute_realtime_combined_pipeline(uniprot_input, target_smiles, radius, n_bits, threshold, fp_type, metric)
    
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
    
    def _execute_realtime_heteroatom_extraction(self, uniprot_input: str):
        """Execute REAL-TIME heteroatom extraction from PDB Data Bank"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        st.info("🌐 Connecting to PDB Data Bank for real-time extraction...")
        
        try:
            # Use real-time extractor
            results_df = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
            
            if not results_df.empty:
                st.success("✅ Real-time heteroatom extraction completed!")
                self._display_heteroatom_results(results_df, "real-time")
            else:
                st.warning("⚠️ No heteroatoms found for provided UniProt IDs")
                
        except Exception as e:
            st.error(f"❌ Real-time heteroatom extraction failed: {str(e)}")
    
    def _execute_realtime_combined_pipeline(self, uniprot_input: str, target_smiles: str,
                                          radius: int, n_bits: int, threshold: float, 
                                          fp_type: str, metric: str):
        """Execute REAL-TIME combined pipeline from PDB Data Bank"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        with st.spinner("🌐 Executing real-time combined pipeline from PDB Data Bank..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Real-time heteroatom extraction
                status_text.text("Step 1/2: Real-time heteroatom extraction from PDB Data Bank...")
                progress_bar.progress(25)
                
                heteroatom_results = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
                progress_bar.progress(50)
                
                # Step 2: Real-time similarity analysis using extracted data
                status_text.text("Step 2/2: Real-time molecular similarity analysis...")
                progress_bar.progress(75)
                
                # Configure real-time similarity analyzer
                self.realtime_similarity.radius = radius
                self.realtime_similarity.n_bits = n_bits
                
                similarity_results = self.realtime_similarity.analyze_similarity_realtime(
                    target_smiles=target_smiles,
                    heteroatom_df=heteroatom_results,
                    min_similarity=threshold
                )
                
                progress_bar.progress(100)
                
                st.success("✅ Real-time combined pipeline completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if not heteroatom_results.empty:
                        st.subheader("🔬 Real-time Heteroatom Results")
                        self._display_heteroatom_results(heteroatom_results, "real-time")
                
                with col2:
                    if not similarity_results.empty:
                        st.subheader("🎯 Real-time Similarity Analysis Results")
                        self._display_similarity_results(similarity_results, target_smiles, "real-time")
                
            except Exception as e:
                st.error(f"❌ Real-time combined pipeline failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
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
        
        # Handle analysis execution based on conversation state
        if "smiles" in st.session_state.current_analysis_state:
            analysis_type = self._extract_analysis_type(user_input)
            threshold = self._extract_threshold_from_text(user_input)
            
            if analysis_type and threshold is not None:
                return self._execute_ai_guided_hybrid_analysis(
                    st.session_state.current_analysis_state["smiles"],
                    analysis_type,
                    threshold,
                    uniprot_ids,
                    st.session_state.current_analysis_state.get("use_local_db", True),
                    st.session_state.current_analysis_state.get("use_realtime", True)
                )
        
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
    
    def _execute_ai_guided_hybrid_analysis(self, smiles: str, analysis_type: str, 
                                          threshold: float, uniprot_ids: List[str] = None,
                                          use_local_db: bool = True, use_realtime: bool = True) -> str:
        """Execute guided analysis using HYBRID data sources"""
        try:
            # Combine data from both sources
            combined_hetero_results = pd.DataFrame()
            combined_similarity_results = pd.DataFrame()
            
            if analysis_type == "heteroatom":
                if not uniprot_ids:
                    return "❌ Heteroatom extraction requires UniProt IDs. Please provide them."
                
                # Local database search
                if use_local_db:
                    local_results = self.local_database[self.local_database['UniProt_ID'].isin(uniprot_ids)]
                    if not local_results.empty:
                        st.subheader("🔬 Local Database Heteroatom Results")
                        self._display_heteroatom_results(local_results, "local-db")
                        combined_hetero_results = pd.concat([combined_hetero_results, local_results], ignore_index=True)
                
                # Real-time fetching
                if use_realtime:
                    realtime_results = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
                    if not realtime_results.empty:
                        st.subheader("🌐 Real-Time Heteroatom Results")
                        self._display_heteroatom_results(realtime_results, "real-time")
                        combined_hetero_results = pd.concat([combined_hetero_results, realtime_results], ignore_index=True)
                
                return f"✅ **Hybrid heteroatom analysis complete!** Found {len(combined_hetero_results)} records total."
            
            elif analysis_type == "similarity":
                # Local database similarity
                if use_local_db:
                    from .similarity_analyzer import SimilarityAnalyzer
                    analyzer = SimilarityAnalyzer()
                    
                    local_sim_results = analyzer.analyze_similarity(
                        target_smiles=smiles,
                        heteroatom_df=self.local_database,
                        min_similarity=threshold
                    )
                    
                    if not local_sim_results.empty:
                        st.subheader("🎯 Local Database Similarity Results")
                        self._display_similarity_results(local_sim_results, smiles, "local-db")
                        combined_similarity_results = pd.concat([combined_similarity_results, local_sim_results], ignore_index=True)
                
                # Real-time similarity (if UniProt IDs available)
                if use_realtime and uniprot_ids:
                    # First get real-time heteroatom data
                    realtime_hetero = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
                    if not realtime_hetero.empty:
                        realtime_sim_results = self.realtime_similarity.analyze_similarity_realtime(
                            target_smiles=smiles,
                            heteroatom_df=realtime_hetero,
                            min_similarity=threshold
                        )
                        if not realtime_sim_results.empty:
                            st.subheader("🌐 Real-Time Similarity Results")
                            self._display_similarity_results(realtime_sim_results, smiles, "real-time")
                            combined_similarity_results = pd.concat([combined_similarity_results, realtime_sim_results], ignore_index=True)
                
                return f"✅ **Hybrid similarity analysis complete!** Found {len(combined_similarity_results)} similar compounds total."
            
            else:  # complete pipeline
                # Combined hybrid analysis
                if uniprot_ids:
                    # Local heteroatom search
                    if use_local_db:
                        local_hetero = self.local_database[self.local_database['UniProt_ID'].isin(uniprot_ids)]
                        combined_hetero_results = pd.concat([combined_hetero_results, local_hetero], ignore_index=True)
                    
                    # Real-time heteroatom extraction
                    if use_realtime:
                        realtime_hetero = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
                        combined_hetero_results = pd.concat([combined_hetero_results, realtime_hetero], ignore_index=True)
                
                # Similarity analysis on combined data
                if use_local_db:
                    from .similarity_analyzer import SimilarityAnalyzer
                    analyzer = SimilarityAnalyzer()
                    local_sim = analyzer.analyze_similarity(
                        target_smiles=smiles,
                        heteroatom_df=self.local_database,
                        min_similarity=threshold
                    )
                    combined_similarity_results = pd.concat([combined_similarity_results, local_sim], ignore_index=True)
                
                if use_realtime and not combined_hetero_results.empty:
                    realtime_sim = self.realtime_similarity.analyze_similarity_realtime(
                        target_smiles=smiles,
                        heteroatom_df=combined_hetero_results,
                        min_similarity=threshold
                    )
                    combined_similarity_results = pd.concat([combined_similarity_results, realtime_sim], ignore_index=True)
                
                # Display hybrid results
                col1, col2 = st.columns(2)
                with col1:
                    if not combined_hetero_results.empty:
                        st.subheader("🔬 Hybrid Heteroatom Results")
                        self._display_heteroatom_results(combined_hetero_results, "hybrid")
                
                with col2:
                    if not combined_similarity_results.empty:
                        st.subheader("🎯 Hybrid Similarity Results")
                        self._display_similarity_results(combined_similarity_results, smiles, "hybrid")
                
                return f"✅ **Complete hybrid pipeline analysis finished!** Heteroatoms: {len(combined_hetero_results)}, Similar compounds: {len(combined_similarity_results)}"
                
        except Exception as e:
            return f"❌ **Hybrid Analysis Error:** {str(e)}\n\nPlease try again or adjust your parameters."
    
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
                        
                        realtime_hetero_results = self.realtime_extractor.extract_heteroatoms_realtime(uniprot_ids)
                        combined_heteroatom_results = pd.concat([combined_heteroatom_results, realtime_hetero_results], ignore_index=True)
                
                # Step 4: Autonomous similarity analysis (HYBRID)
                status_text.text("🧪 AI performing autonomous similarity analysis (hybrid)...")
                progress_bar.progress(70)
                
                if smiles:
                    # Local similarity
                    if use_local_data:
                        from .similarity_analyzer import SimilarityAnalyzer
                        analyzer = SimilarityAnalyzer(**ai_params)
                        
                        local_similarity_results = analyzer.analyze_similarity(
                            target_smiles=smiles,
                            heteroatom_df=self.local_database,
                            min_similarity=ai_params["threshold"]
                        )
                        combined_similarity_results = pd.concat([combined_similarity_results, local_similarity_results], ignore_index=True)
                    
                    # Real-time similarity
                    if use_realtime_data and not combined_heteroatom_results.empty:
                        realtime_similarity_results = self.realtime_similarity.analyze_similarity_realtime(
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
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Top compounds bar chart
            top_10 = similarity_results.head(10)
            fig_bar = px.bar(
                top_10,
                x='Heteroatom_Code',
                y='Tanimoto_Similarity',
                title="Top 10 Similar Compounds"
            )
            fig_bar.update_xaxes(tickangle=45)
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
    
    def _extract_analysis_type(self, text: str) -> str:
        """Extract analysis type from text"""
        text_lower = text.lower()
        
        if 'heteroatom' in text_lower and ('similarity' in text_lower or 'complete' in text_lower or 'pipeline' in text_lower):
            return 'complete'
        elif 'heteroatom' in text_lower:
            return 'heteroatom'
        elif 'similarity' in text_lower:
            return 'similarity'
        elif 'complete' in text_lower or 'pipeline' in text_lower or 'both' in text_lower:
            return 'complete'
        
        return ''
    
    def _extract_threshold_from_text(self, text: str) -> Optional[float]:
        """Extract threshold value from text"""
        import re
        
        # Look for decimal numbers
        numbers = re.findall(r'0\.\d+', text)
        for num in numbers:
            val = float(num)
            if 0 <= val <= 1:
                return val
        
        # Look for keywords
        if '0.5' in text or 'loose' in text.lower():
            return 0.5
        elif '0.7' in text or 'moderate' in text.lower():
            return 0.7
        elif '0.9' in text or 'strict' in text.lower():
            return 0.9
        
        return None

    def _display_heteroatom_results(self, results: pd.DataFrame, mode: str):
        """Display heteroatom extraction results"""
        if results.empty:
            st.info("No heteroatom results to display")
            return
        
        # Filter out NO_HETEROATOMS entries
        valid_results = results[results['Heteroatom_Code'] != 'NO_HETEROATOMS']
        
        if not valid_results.empty:
            st.dataframe(valid_results[['UniProt_ID', 'PDB_ID', 'Heteroatom_Code', 'SMILES', 'Chemical_Name', 'Formula']].head(10))
            
            # Download option
            csv = valid_results.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Heteroatom Results ({mode})",
                data=csv,
                file_name=f"heteroatom_results_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No valid heteroatom records found")
    
    def _display_similarity_results(self, results: pd.DataFrame, target_smiles: str, mode: str):
        """Display similarity analysis results"""
        if results.empty:
            st.info("No similarity results to display")
            return
        
        # Display top results
        display_cols = ['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'SMILES', 'Tanimoto_Similarity', 'Formula']
        available_cols = [col for col in display_cols if col in results.columns]
        
        st.dataframe(results[available_cols].head(10))
        
        # Create visualization
        if len(results) > 5 and 'Tanimoto_Similarity' in results.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(results.head(20)))),
                y=results.head(20)['Tanimoto_Similarity'],
                mode='markers+lines',
                name='Similarity Score',
                marker=dict(size=8, color=results.head(20)['Tanimoto_Similarity'], colorscale='Viridis')
            ))
            fig.update_layout(title=f"Top 20 Similarity Scores ({mode} mode)", xaxis_title="Compound Rank", yaxis_title="Tanimoto Similarity")
            st.plotly_chart(fig, use_container_width=True)
        
        # Download option
        csv = results.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Similarity Results ({mode})",
            data=csv,
            file_name=f"similarity_results_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )