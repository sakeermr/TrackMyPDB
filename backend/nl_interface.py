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
        
        # Compact header with key info only
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🎯 Processing {len(uniprot_ids)} UniProt IDs")
        with col2:
            st.info("📋 Parallel downloads enabled")
        with col3:
            st.info("⚡ Batch processing active")

        total_start_time = time.time()
        
        # Single progress container for all processing
        progress_container = st.container()
        with progress_container:
            overall_progress = st.progress(0)
            status_display = st.empty()
            
            # Horizontal status indicators
            status_cols = st.columns(4)
            with status_cols[0]:
                current_uniprot = st.empty()
            with status_cols[1]:
                pdb_status = st.empty()
            with status_cols[2]:
                processing_status = st.empty()
            with status_cols[3]:
                file_counter = st.empty()
        
        total_progress = 0
        total_steps = len(uniprot_ids) * 3  # 3 steps per UniProt: fetch, download, process
        
        for idx, uniprot_id in enumerate(uniprot_ids):
            # Update current UniProt being processed
            current_uniprot.metric("🧬 Current UniProt", uniprot_id)
            
            # Step 1: Get PDB IDs
            pdb_status.info("🔍 Fetching PDB IDs...")
            pdb_ids = self.get_pdbs_for_uniprot(uniprot_id)
            total_progress += 1
            overall_progress.progress(total_progress / total_steps)
            
            if not pdb_ids:
                pdb_status.warning("⚠️ No PDBs found")
                total_progress += 2  # Skip remaining steps
                overall_progress.progress(total_progress / total_steps)
                continue
            
            pdb_status.success(f"✅ Found {len(pdb_ids)} PDBs")
            
            # Step 2: Download PDB files
            processing_status.info("⬇️ Downloading...")
            file_counter.metric("📁 Files", f"0/{len(pdb_ids)}")
            
            pdb_data = self.download_pdb_parallel_compact(pdb_ids, max_workers=5, file_counter=file_counter)
            total_progress += 1
            overall_progress.progress(total_progress / total_steps)
            
            if not pdb_data:
                processing_status.warning("⚠️ Download failed")
                total_progress += 1
                overall_progress.progress(total_progress / total_steps)
                continue
            
            processing_status.success(f"✅ Downloaded {len(pdb_data)} files")
            
            # Step 3: Process heteroatoms
            file_counter.metric("🔬 Processing", f"{len(pdb_data)} files")
            uniprot_results = self.process_pdb_heteroatoms_compact(pdb_data, uniprot_id)
            self.all_records.extend(uniprot_results)
            total_progress += 1
            overall_progress.progress(total_progress / total_steps)
            
            # Update final status for this UniProt
            heteroatom_count = len([r for r in uniprot_results if r['Heteroatom_Code'] != 'NO_HETEROATOMS'])
            processing_status.success(f"✅ Found {heteroatom_count} heteroatoms")

        # Clear progress indicators
        progress_container.empty()
        
        # Create final DataFrame
        df = pd.DataFrame(self.all_records)
        total_time = time.time() - total_start_time
        
        # Compact results summary
        st.success(f"🎉 EXTRACTION COMPLETE in {total_time:.1f}s!")
        
        # Horizontal metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Records", len(df))
        with col2:
            st.metric("🏗️ PDBs", df['PDB_ID'].nunique())
        with col3:
            st.metric("🧪 Heteroatoms", df['Heteroatom_Code'].nunique())
        with col4:
            st.metric("✅ With SMILES", len(df[df['SMILES'] != '']))

        return df

    def download_pdb_parallel_compact(self, pdb_ids, max_workers=5, file_counter=None):
        """Download multiple PDB files in parallel - COMPACT DISPLAY"""
        pdb_data = {}
        
        def download_single_pdb(pdb_id):
            try:
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                r = self.session.get(url, timeout=45)
                r.raise_for_status()
                return pdb_id, r.text.splitlines()
            except Exception:
                return pdb_id, None
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                except Exception:
                    self.failed_pdbs.append(pdb_id)
                
                completed += 1
                # Update file counter compactly
                if file_counter:
                    file_counter.metric("📁 Files", f"{completed}/{len(pdb_ids)}")
        
        return pdb_data

    def process_pdb_heteroatoms_compact(self, pdb_data, uniprot_id):
        """Process heteroatoms from multiple PDBs - COMPACT DISPLAY"""
        all_results = []
        
        for pdb_id, lines in pdb_data.items():
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
                continue

            # Process each heteroatom code
            for code in codes:
                details = het_details.get(code, {})
                chains = ', '.join(sorted(details.get('chains', set())))
                res_nums = ', '.join(sorted(details.get('residue_numbers', set())))
                atom_count = len(details.get('atom_names', set()))

                # Fetch SMILES (optimized)
                rcsb_data = self.fetch_smiles_rcsb_optimized(code)
                smiles = rcsb_data['smiles']

                # Try PubChem backup if no SMILES
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

                # Minimal delay
                time.sleep(0.05)
        
        return all_results

    def analyze_similarity_realtime_optimized(self, target_smiles, heteroatom_df, top_n=50, min_similarity=0.01):
        """
        Perform OPTIMIZED real-time similarity analysis - COMPACT DISPLAY
        """
        try:
            # Import RDKit for real-time analysis
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors, DataStructs
            import numpy as np
            
            # Compact header
            st.info("🔄 OPTIMIZED molecular similarity analysis starting...")
            
            # Filter out rows without SMILES - OPTIMIZED
            valid_df = heteroatom_df[
                (heteroatom_df['SMILES'].notna()) & 
                (heteroatom_df['SMILES'] != '') & 
                (heteroatom_df['Heteroatom_Code'] != 'NO_HETEROATOMS')
            ].copy()
            
            # Compact status display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 {len(valid_df)} valid SMILES")
            with col2:
                st.info(f"🎯 Target: {target_smiles[:15]}...")
            with col3:
                st.info(f"⚡ Threshold: {min_similarity}")

            if len(valid_df) == 0:
                st.warning("⚠️ No valid SMILES found for similarity analysis")
                return pd.DataFrame()

            # Single progress container for compact display
            progress_container = st.container()
            with progress_container:
                overall_progress = st.progress(0)
                
                # Horizontal status indicators
                status_cols = st.columns(4)
                with status_cols[0]:
                    fp_status = st.empty()
                with status_cols[1]:
                    batch_counter = st.empty()
                with status_cols[2]:
                    similarity_status = st.empty()
                with status_cols[3]:
                    result_counter = st.empty()

            # OPTIMIZED fingerprint computation
            fp_status.info("🧮 Computing fingerprints...")
            fingerprints = []
            valid_indices = []
            
            batch_size = 50  # Process in batches for better performance
            total_batches = (len(valid_df) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(valid_df))
                batch = valid_df.iloc[start_idx:end_idx]
                
                batch_counter.metric("📦 Batch", f"{batch_idx + 1}/{total_batches}")
                
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
                
                # Update progress (first half)
                progress = (batch_idx + 1) / (total_batches * 2)
                overall_progress.progress(progress)

            processed_df = valid_df.loc[valid_indices].copy()
            processed_df['Fingerprint'] = fingerprints

            fp_status.success(f"✅ Processed {len(processed_df)} molecules")
            
            # OPTIMIZED similarity calculation
            similarity_status.info("🎯 Calculating similarities...")

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
                overall_progress.progress(progress)
                
                # Update batch counter
                batch_counter.metric("📦 Similarity Batch", f"{batch_idx + 1}/{total_batches}")

            # Clear progress container
            progress_container.empty()

            # Add similarity scores to DataFrame
            result_df = processed_df.copy()
            result_df['Tanimoto_Similarity'] = similarities

            # Filter by minimum similarity and sort - OPTIMIZED
            result_df = result_df[result_df['Tanimoto_Similarity'] >= min_similarity]
            result_df = result_df.sort_values('Tanimoto_Similarity', ascending=False)

            # Return top results
            top_results = result_df.head(top_n)

            # Compact results summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏆 Found", len(result_df))
            with col2:
                st.metric("🔝 Top", len(top_results))
            with col3:
                avg_sim = result_df['Tanimoto_Similarity'].mean() if len(result_df) > 0 else 0
                st.metric("📊 Avg Score", f"{avg_sim:.3f}")
            with col4:
                max_sim = result_df['Tanimoto_Similarity'].max() if len(result_df) > 0 else 0
                st.metric("🎯 Best Score", f"{max_sim:.3f}")

            return top_results
            
        except ImportError:
            st.error("❌ RDKit not available for real-time similarity analysis")
            st.info("💡 Please install RDKit: `pip install rdkit`")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error in optimized similarity analysis: {str(e)}")
            return pd.DataFrame()
    
    def _display_heteroatom_results(self, results: pd.DataFrame, mode: str):
        """Display heteroatom extraction results with compact horizontal formatting"""
        if results.empty:
            st.info("No heteroatom results to display")
            return
        
        # Filter out NO_HETEROATOMS entries
        valid_results = results[results['Heteroatom_Code'] != 'NO_HETEROATOMS']
        
        if not valid_results.empty:
            # Compact horizontal metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📊 Total", len(valid_results))
            with col2:
                st.metric("🧪 Unique", valid_results['Heteroatom_Code'].nunique())
            with col3:
                st.metric("✅ SMILES", len(valid_results[valid_results['SMILES'] != '']))
            with col4:
                st.metric("🏗️ PDBs", valid_results['PDB_ID'].nunique())
            with col5:
                success_rate = (len(valid_results[valid_results['SMILES'] != '']) / len(valid_results) * 100) if len(valid_results) > 0 else 0
                st.metric("📈 Success", f"{success_rate:.0f}%")
            
            # Compact table display - show only top 10 to prevent vertical expansion
            st.subheader("🔝 Top 10 Results")
            display_cols = ['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'SMILES', 'Status']
            available_cols = [col for col in display_cols if col in valid_results.columns]
            st.dataframe(valid_results[available_cols].head(10), use_container_width=True, height=300)
            
            # Compact download section - horizontal buttons
            st.subheader("📥 Downloads")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = valid_results.to_csv(index=False)
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name=f"heteroatom_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                json_data = valid_results.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 JSON",
                    data=json_data,
                    file_name=f"heteroatom_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col3:
                summary = self._generate_heteroatom_summary(valid_results, mode)
                st.download_button(
                    label="📄 Summary",
                    data=summary,
                    file_name=f"heteroatom_summary_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("No valid heteroatom records found")
    
    def _display_similarity_results(self, results: pd.DataFrame, target_smiles: str, mode: str):
        """Display similarity analysis results with compact horizontal layout"""
        if results.empty:
            st.info("No similarity results to display")
            return
        
        # Compact horizontal metrics
        if 'Tanimoto_Similarity' in results.columns:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🎯 Matches", len(results))
            with col2:
                st.metric("🏆 Best", f"{results['Tanimoto_Similarity'].max():.3f}")
            with col3:
                st.metric("📊 Average", f"{results['Tanimoto_Similarity'].mean():.3f}")
            with col4:
                st.metric("🧪 PDBs", results['PDB_ID'].nunique())
            with col5:
                high_sim = len(results[results['Tanimoto_Similarity'] >= 0.8])
                st.metric("🔥 High (≥0.8)", high_sim)
        
        # Compact table display - show only top 15 to prevent vertical expansion
        st.subheader("🔝 Top 15 Similar Compounds")
        display_cols = ['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'Tanimoto_Similarity', 'SMILES']
        available_cols = [col for col in display_cols if col in results.columns]
        st.dataframe(results[available_cols].head(15), use_container_width=True, height=400)
        
        # Compact visualizations - side by side to save vertical space
        if len(results) > 5:
            st.subheader("📊 Analysis Charts")
            col1, col2 = st.columns(2)
            
            with col1:
                # Compact histogram
                fig_hist = px.histogram(
                    results.head(50),  # Limit data for performance
                    x='Tanimoto_Similarity',
                    title="Similarity Distribution",
                    nbins=15,
                    height=300,  # Reduced height
                    color_discrete_sequence=['#00cc96']
                )
                fig_hist.update_layout(margin=dict(t=30, b=30, l=30, r=30))
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Compact scatter plot
                top_15 = results.head(15)
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=list(range(len(top_15))),
                    y=top_15['Tanimoto_Similarity'],
                    mode='markers+lines',
                    name='Similarity Score',
                    marker=dict(size=8, color=top_15['Tanimoto_Similarity'], colorscale='Viridis'),
                    text=[f"{row['PDB_ID']}-{row['Heteroatom_Code']}" for _, row in top_15.iterrows()],
                    hovertemplate='<b>%{text}</b><br>Similarity: %{y:.3f}<extra></extra>'
                ))
                fig_scatter.update_layout(
                    title="Top 15 Similarity Scores", 
                    xaxis_title="Rank", 
                    yaxis_title="Similarity",
                    height=300,  # Reduced height
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Compact download section - horizontal buttons
        st.subheader("📥 Downloads")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 CSV Results",
                data=csv,
                file_name=f"similarity_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            summary_report = self._generate_similarity_summary(results, target_smiles, mode)
            st.download_button(
                label="📄 Report",
                data=summary_report,
                file_name=f"similarity_report_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col3:
            # Generate insights
            insights = self._generate_similarity_insights(results, target_smiles)
            st.download_button(
                label="💡 Insights",
                data=insights,
                file_name=f"similarity_insights_{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    def _generate_heteroatom_summary(self, results: pd.DataFrame, mode: str) -> str:
        """Generate compact summary for heteroatom results"""
        summary = f"""HETEROATOM EXTRACTION SUMMARY ({mode.upper()})
==================================================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: {mode}

OVERVIEW:
- Total Records: {len(results)}
- Unique PDBs: {results['PDB_ID'].nunique()}
- Unique Heteroatoms: {results['Heteroatom_Code'].nunique()}
- Records with SMILES: {len(results[results['SMILES'] != ''])}

TOP 10 HETEROATOMS:
"""
        top_heteroatoms = results['Heteroatom_Code'].value_counts().head(10)
        for code, count in top_heteroatoms.items():
            summary += f"  {code}: {count} occurrences\n"
        
        return summary

    def _generate_similarity_insights(self, results: pd.DataFrame, target_smiles: str) -> str:
        """Generate insights for similarity analysis"""
        if 'Tanimoto_Similarity' not in results.columns:
            return "No similarity data available for insights."
        
        insights = f"""SIMILARITY ANALYSIS INSIGHTS
============================
Target: {target_smiles}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY INSIGHTS:
- Best Match: {results.iloc[0]['PDB_ID']}-{results.iloc[0]['Heteroatom_Code']} (Score: {results.iloc[0]['Tanimoto_Similarity']:.3f})
- High Similarity Compounds (≥0.8): {len(results[results['Tanimoto_Similarity'] >= 0.8])}
- Medium Similarity Compounds (0.5-0.8): {len(results[(results['Tanimoto_Similarity'] >= 0.5) & (results['Tanimoto_Similarity'] < 0.8)])}
- Unique PDB Structures: {results['PDB_ID'].nunique()}

RECOMMENDATIONS:
- Focus on compounds with similarity ≥ 0.7 for structural analysis
- Consider PDB structures with multiple similar heteroatoms
- Validate top matches through experimental studies
"""
        return insights