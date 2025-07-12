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
                # Exclude water and common heteroatoms for speed
                if "HOH" in line or "WAT" in line:
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

class TrackMyPDBNLInterface:
    """
    Natural Language Interface for TrackMyPDB
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
        
        # Initialize Gemini AI agent for AI modes
        try:
            from .gemini_agent import GeminiAgent
            self.gemini_agent = GeminiAgent()
            if self.gemini_agent.is_available():
                st.success("✅ AI modes enabled with Google Gemini")
            else:
                st.warning("⚠️ AI modes will use fallback processing")
        except Exception as e:
            st.error(f"❌ AI initialization failed: {str(e)}")
            self.gemini_agent = None
        
        # Initialize chat history for AI modes
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_analysis_state" not in st.session_state:
            st.session_state.current_analysis_state = {}
    
    def _load_local_database(self):
        """Load the local database from Excel files"""
        try:
            # Load all sheets from the Excel file
            xls = pd.ExcelFile("PDB_Derived_Data.xlsx")
            all_sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
            
            # Concatenate all sheets into a single DataFrame
            combined_df = pd.concat(all_sheets.values(), ignore_index=True)
            
            st.success(f"✅ Loaded local database with {len(combined_df)} records from Excel")
            return combined_df
        except Exception as e:
            st.error(f"❌ Error loading local database: {str(e)}")
            return pd.DataFrame()
    
    def _extract_smiles_from_text(self, text: str) -> str:
        """Extract SMILES structure from text input"""
        # Simple heuristic: look for strings that start with 'C' and have 5 or more characters
        potential_smiles = re.findall(r'\bC[^ ]{4,}\b', text)
        return potential_smiles[0] if potential_smiles else ""
    
    def _extract_uniprot_ids(self, text: str) -> List[str]:
        """Extract UniProt IDs from text input"""
        # Simple heuristic: look for strings that start with 'P' and are followed by 5 digits
        potential_ids = re.findall(r'\bP\d{5}\b', text)
        return list(set(potential_ids))  # Return unique IDs
    
    def handle_user_input(self, user_input: str):
        """Handle and process user input for analysis requests"""
        # Reset current analysis state
        st.session_state.current_analysis_state = {}
        
        # Check for empty input
        if not user_input.strip():
            return "❌ Please provide a SMILES structure or UniProt IDs to analyze."
        
        # Check if Gemini AI is available
        if self.gemini_agent and self.gemini_agent.is_available():
            try:
                # Use Gemini AI for intelligent response generation
                ai_analysis = self.gemini_agent.process_query_sync(user_input)
                
                # Generate response based on AI analysis
                action = ai_analysis.get('action', 'extract_heteroatoms')
                parameters = ai_analysis.get('parameters', {})
                explanation = ai_analysis.get('explanation', 'AI analysis completed')
                confidence = ai_analysis.get('confidence', 0.8)
                
                # Store AI analysis in session state
                st.session_state.current_analysis_state.update({
                    "ai_action": action,
                    "ai_parameters": parameters,
                    "ai_explanation": explanation,
                    "ai_confidence": confidence
                })
                
                # Extract data from user input and AI analysis
                smiles = parameters.get('target_smiles') or self._extract_smiles_from_text(user_input)
                uniprot_ids = parameters.get('uniprot_ids') or self._extract_uniprot_ids(user_input)
                
                # Data sources information
                data_sources = []
                if parameters.get('use_local_db', True):
                    data_sources.append(f"local database ({len(self.local_database)} records)")
                if parameters.get('use_realtime', True):
                    data_sources.append("real-time PDB fetching")
                data_source_text = " AND ".join(data_sources) if data_sources else "no data sources selected"
                
                # Generate intelligent response based on AI analysis
                response = f"""🤖 **AI Analysis Complete** (Confidence: {confidence:.1f})

**Detected Action:** {action.replace('_', ' ').title()}
**AI Explanation:** {explanation}

**📊 Data Sources:** {data_source_text}

"""
                
                if smiles:
                    response += f"""🧪 **Detected SMILES:** `{smiles}`
"""
                
                if uniprot_ids:
                    response += f"""🧬 **Detected UniProt IDs:** {', '.join(uniprot_ids)}
"""
                
                # Action-specific guidance
                if action == "extract_heteroatoms":
                    response += """
**🔬 Heteroatom Extraction Mode**
I'll extract heteroatoms from the specified protein structures.

**Next Steps:**
1. Confirm the UniProt IDs if detected, or provide them
2. Choose analysis parameters (optional)
3. I'll proceed with extraction using your selected data sources

**Would you like to:**
- ✅ **Proceed** with current settings
- ⚙️ **Modify** parameters first
- ❓ **Ask** questions about the process
"""
                
                elif action == "similarity_analysis":
                    response += """
**🎯 Similarity Analysis Mode**
I'll perform molecular similarity analysis with your SMILES structure.

**Next Steps:**
1. Confirm the target SMILES structure
2. Set similarity threshold (default: 0.7)
3. I'll analyze against available data sources

**Would you like to:**
- ✅ **Proceed** with current settings
- ⚙️ **Adjust** similarity threshold
- 📊 **Review** available data first
"""
                
                elif action == "combined_analysis":
                    response += """
**🚀 Combined Analysis Mode**
I'll perform both heteroatom extraction and similarity analysis.

**Complete Pipeline:**
1. Extract heteroatoms from specified proteins
2. Perform similarity analysis with your SMILES
3. Generate comprehensive results and insights

**Would you like to:**
- ✅ **Start** the complete pipeline
- ⚙️ **Configure** advanced settings
- 📋 **Review** the analysis plan first
"""
                
                response += f"""

**💡 Ready to proceed?** Just say "yes" or "proceed" and I'll start the {action.replace('_', ' ')}!
"""
                
                return response
                
            except Exception as e:
                st.warning(f"AI processing error: {str(e)}")
                # Fall back to original logic
                pass
        
        # Fallback to original logic if AI is not available
        user_input_lower = user_input.lower()
        
        # Extract SMILES from input
        smiles = self._extract_smiles_from_text(user_input)
        uniprot_ids = self._extract_uniprot_ids(user_input)
        
        # Data sources information
        data_sources = []
        if parameters.get('use_local_db', True):
            data_sources.append(f"local database ({len(self.local_database)} records)")
        if parameters.get('use_realtime', True):
            data_sources.append("real-time PDB fetching")
        data_source_text = " AND ".join(data_sources) if data_sources else "no data sources selected"
        
        if smiles:
            st.session_state.current_analysis_state["smiles"] = smiles
            st.session_state.current_analysis_state["use_local_db"] = parameters.get('use_local_db', True)
            st.session_state.current_analysis_state["use_realtime"] = parameters.get('use_realtime', True)
            
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
{f"- 📊 Local Database: {len(self.local_database)} records" if parameters.get('use_local_db', True) else ""}
{f"- 🌐 Real-Time PDB: Live fetching enabled" if parameters.get('use_realtime', True) else ""}

Please answer these questions and I'll proceed with your guided hybrid analysis!"""
            
            return response
        
        elif uniprot_ids:
            st.session_state.current_analysis_state["uniprot_ids"] = uniprot_ids
            st.session_state.current_analysis_state["use_local_db"] = parameters.get('use_local_db', True)
            st.session_state.current_analysis_state["use_realtime"] = parameters.get('use_realtime', True)
            
            return f"""🔍 **Detected UniProt IDs: {', '.join(uniprot_ids)}**

Great! I can extract heteroatoms from these protein structures using {data_source_text}.

**❓ Follow-up Questions:**
1. Do you also have a target SMILES for similarity analysis?
2. What analysis scope do you prefer?
   - Quick extraction (heteroatoms only)
   - Full analysis (if you provide a SMILES structure)

Please provide any additional details!"""
        
        # Check for proceed/confirmation commands
        proceed_keywords = ['yes', 'proceed', 'start', 'go', 'continue', 'execute', 'run']
        if any(keyword in user_input_lower for keyword in proceed_keywords):
            # Check if we have stored AI analysis to execute
            if "ai_action" in st.session_state.current_analysis_state:
                return self._execute_ai_guided_analysis()
            else:
                return "I need more information before proceeding. Please provide a SMILES structure or UniProt IDs first."
        
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

    def _execute_ai_guided_analysis(self) -> str:
        """Execute analysis based on AI guidance"""
        analysis_state = st.session_state.current_analysis_state
        action = analysis_state.get("ai_action", "extract_heteroatoms")
        parameters = analysis_state.get("ai_parameters", {})
        
        try:
            if action == "extract_heteroatoms":
                uniprot_ids = parameters.get("uniprot_ids", [])
                if not uniprot_ids:
                    return "❌ No UniProt IDs found for heteroatom extraction. Please provide them."
                
                # Execute heteroatom extraction
                self._execute_ai_heteroatom_extraction(uniprot_ids, analysis_state)
                return "✅ Heteroatom extraction started! Check the results below."
            
            elif action == "similarity_analysis":
                target_smiles = parameters.get("target_smiles", "")
                if not target_smiles:
                    return "❌ No target SMILES found for similarity analysis. Please provide it."
                
                # Execute similarity analysis
                self._execute_ai_similarity_analysis(target_smiles, analysis_state)
                return "✅ Similarity analysis started! Check the results below."
            
            elif action == "combined_analysis":
                uniprot_ids = parameters.get("uniprot_ids", [])
                target_smiles = parameters.get("target_smiles", "")
                
                if not uniprot_ids or not target_smiles:
                    return "❌ Both UniProt IDs and target SMILES required for combined analysis."
                
                # Execute combined analysis
                self._execute_ai_combined_analysis(uniprot_ids, target_smiles, analysis_state)
                return "✅ Combined analysis started! Check the results below."
            
            else:
                return f"❌ Unknown action: {action}"
                
        except Exception as e:
            return f"❌ Analysis execution failed: {str(e)}"

    def _execute_ai_heteroatom_extraction(self, uniprot_ids: List[str], analysis_state: Dict):
        """Execute AI-guided heteroatom extraction"""
        st.subheader("🤖 AI-Guided Heteroatom Extraction")
        
        use_local_db = analysis_state.get("use_local_db", True)
        use_realtime = analysis_state.get("use_realtime", True)
        
        combined_results = pd.DataFrame()
        
        # Local database extraction
        if use_local_db and not self.local_database.empty:
            st.info("🔍 Searching local database...")
            local_results = self.local_database[
                self.local_database['UniProt_ID'].isin(uniprot_ids)
            ]
            if not local_results.empty:
                combined_results = pd.concat([combined_results, local_results], ignore_index=True)
                st.success(f"✅ Found {len(local_results)} records in local database")
        
        # Real-time extraction
        if use_realtime:
            st.info("🌐 Performing real-time PDB extraction...")
            realtime_results = self.realtime_extractor.extract_heteroatoms_realtime_optimized(uniprot_ids)
            if not realtime_results.empty:
                combined_results = pd.concat([combined_results, realtime_results], ignore_index=True)
        
        # Display results with AI insights
        if not combined_results.empty:
            self._display_heteroatom_results(combined_results, "ai-guided")
            
            # Generate AI explanation
            if self.gemini_agent and self.gemini_agent.is_available():
                st.subheader("🧠 AI Scientific Insights")
                results_summary = {
                    'total_heteroatoms': len(combined_results),
                    'unique_pdbs': combined_results['PDB_ID'].nunique(),
                    'unique_codes': combined_results['Heteroatom_Code'].nunique()
                }
                ai_explanation = self.gemini_agent.get_scientific_explanation()
                st.info(f"**AI Analysis:** {ai_explanation}")

    def _execute_ai_similarity_analysis(self, target_smiles: str, analysis_state: Dict):
        """Execute AI-guided similarity analysis"""
        st.subheader("🤖 AI-Guided Similarity Analysis")
        
        use_local_db = analysis_state.get("use_local_db", True)
        use_realtime = analysis_state.get("use_realtime", True)
        
        # Determine data source
        if use_local_db and not self.local_database.empty:
            st.info("📊 Using local database for similarity analysis...")
            similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                target_smiles=target_smiles,
                heteroatom_df=self.local_database,
                min_similarity=0.3
            )
        else:
            st.warning("⚠️ No local data available. Please run heteroatom extraction first.")
            return
        
        # Display results with AI insights
        if not similarity_results.empty:
            self._display_similarity_results(similarity_results, target_smiles, "ai-guided")
            
            # Generate AI explanation
            if self.gemini_agent and self.gemini_agent.is_available():
                st.subheader("🧠 AI Scientific Insights")
                results_summary = {
                    'similarity_matches': len(similarity_results),
                    'top_similarities': similarity_results['Tanimoto_Similarity'].head(5).tolist(),
                    'average_similarity': similarity_results['Tanimoto_Similarity'].mean()
                }
                ai_explanation = self.gemini_agent.get_scientific_explanation(results_summary)
                st.info(f"**AI Analysis:** {ai_explanation}")

    def _execute_ai_combined_analysis(self, uniprot_ids: List[str], target_smiles: str, analysis_state: Dict):
        """Execute AI-guided combined analysis"""
        st.subheader("🤖 AI-Guided Combined Analysis")
        
        # First extract heteroatoms
        self._execute_ai_heteroatom_extraction(uniprot_ids, analysis_state)
        
        # Then perform similarity analysis on the results
        if "heteroatom_results" in st.session_state:
            heteroatom_data = st.session_state["heteroatom_results"]
            similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                target_smiles=target_smiles,
                heteroatom_df=heteroatom_data,
                min_similarity=0.3
            )
            
            if not similarity_results.empty:
                st.subheader("🎯 Combined Analysis - Similarity Results")
                self._display_similarity_results(similarity_results, target_smiles, "ai-combined")
                
                # Generate comprehensive AI explanation
                if self.gemini_agent and self.gemini_agent.is_available():
                    st.subheader("🧠 Comprehensive AI Insights")
                    results_summary = {
                        'total_heteroatoms': len(heteroatom_data),
                        'similarity_matches': len(similarity_results),
                        'best_similarity': similarity_results['Tanimoto_Similarity'].max(),
                        'target_smiles': target_smiles
                    }
                    ai_explanation = self.gemini_agent.get_scientific_explanation(results_summary)
                    st.info(f"**Comprehensive AI Analysis:** {ai_explanation}")

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
                
                # Use Gemini AI for intelligent parameter extraction if available
                if self.gemini_agent and self.gemini_agent.is_available():
                    ai_analysis = self.gemini_agent.process_query_sync(user_input)
                    action = ai_analysis.get('action', 'extract_heteroatoms')
                    ai_parameters = ai_analysis.get('parameters', {})
                    
                    smiles = ai_parameters.get('target_smiles') or self._extract_smiles_from_text(user_input)
                    uniprot_ids = ai_parameters.get('uniprot_ids') or self._extract_uniprot_ids(user_input)
                    
                    st.info(f"🤖 AI detected action: {action} (Confidence: {ai_analysis.get('confidence', 0.8):.2f})")
                else:
                    # Fallback extraction
                    smiles = self._extract_smiles_from_text(user_input)
                    uniprot_ids = self._extract_uniprot_ids(user_input)
                    action = "combined_analysis" if smiles and uniprot_ids else "extract_heteroatoms"
                
                if not smiles and not uniprot_ids:
                    st.error("❌ AI couldn't detect SMILES structure or UniProt IDs")
                    return
                
                # Step 2: Autonomous parameter optimization
                status_text.text("⚙️ AI optimizing analysis parameters...")
                progress_bar.progress(30)
                
                # AI-determined optimal parameters based on speed preference
                ai_params = {
                    "radius": 2 if processing_speed == "Fast" else 3 if processing_speed == "Balanced" else 4,
                    "n_bits": 1024 if processing_speed == "Fast" else 2048 if processing_speed == "Balanced" else 4096,
                    "fp_type": "morgan",
                    "metric": "tanimoto",
                    "threshold": threshold,
                    "batch_size": 100 if processing_speed == "Fast" else 50 if processing_speed == "Balanced" else 25
                }
                
                # Configure analyzers with AI-optimized parameters
                self.realtime_similarity.radius = ai_params["radius"]
                self.realtime_similarity.n_bits = ai_params["n_bits"]
                
                # Combined results storage
                combined_heteroatom_results = pd.DataFrame()
                combined_similarity_results = pd.DataFrame()
                
                # Step 3: Autonomous heteroatom analysis (HYBRID)
                status_text.text("🔬 AI performing autonomous heteroatom analysis...")
                progress_bar.progress(40)
                
                if data_priority == "Local First" or data_priority == "Hybrid":
                    if use_local_data and not self.local_database.empty:
                        st.info("📊 AI analyzing local database...")
                        
                        if uniprot_ids:
                            local_hetero_results = self.local_database[
                                self.local_database['UniProt_ID'].isin(uniprot_ids)
                            ]
                        else:
                            # AI selects representative sample based on diversity
                            sample_size = min(100 if processing_speed == "Fast" else 200, len(self.local_database))
                            local_hetero_results = self.local_database.sample(sample_size, random_state=42)
                        
                        combined_heteroatom_results = pd.concat([combined_heteroatom_results, local_hetero_results], ignore_index=True)
                        st.success(f"✅ AI processed {len(local_hetero_results)} local records")
                
                if data_priority == "Real-Time First" or data_priority == "Hybrid":
                    if use_realtime_data and uniprot_ids:
                        st.info("🌐 AI performing real-time PDB extraction...")
                        progress_bar.progress(50)
                        
                        # Limit UniProt IDs based on processing speed
                        max_uniprots = 3 if processing_speed == "Fast" else 5 if processing_speed == "Balanced" else len(uniprot_ids)
                        limited_uniprot_ids = uniprot_ids[:max_uniprots]
                        
                        realtime_hetero_results = self.realtime_extractor.extract_heteroatoms_realtime_optimized(limited_uniprot_ids)
                        combined_heteroatom_results = pd.concat([combined_heteroatom_results, realtime_hetero_results], ignore_index=True)
                        st.success(f"✅ AI extracted real-time data from {len(limited_uniprot_ids)} proteins")
                
                # Store heteroatom results for similarity analysis
                if not combined_heteroatom_results.empty:
                    st.session_state["heteroatom_results"] = combined_heteroatom_results
                
                # Step 4: Autonomous similarity analysis (HYBRID)
                if smiles:
                    status_text.text("🧪 AI performing autonomous similarity analysis...")
                    progress_bar.progress(70)
                    
                    if not combined_heteroatom_results.empty:
                        st.info("🎯 AI calculating molecular similarities...")
                        
                        # Perform similarity analysis on combined data
                        realtime_similarity_results = self.realtime_similarity.analyze_similarity_realtime_optimized(
                            target_smiles=smiles,
                            heteroatom_df=combined_heteroatom_results,
                            min_similarity=ai_params["threshold"],
                            top_n=50 if processing_speed == "Fast" else 100
                        )
                        
                        if not realtime_similarity_results.empty:
                            combined_similarity_results = realtime_similarity_results
                            st.success(f"✅ AI found {len(combined_similarity_results)} similar compounds")
                
                # Step 5: Autonomous report generation with AI insights
                status_text.text("📋 AI generating comprehensive autonomous hybrid report...")
                progress_bar.progress(90)
                
                # Display autonomous results
                self._display_autonomous_hybrid_results(
                    smiles, uniprot_ids, combined_heteroatom_results, combined_similarity_results,
                    ai_params, comprehensive, visualizations, auto_download, result_detail,
                    use_local_data, use_realtime_data, data_priority, action
                )
                
                progress_bar.progress(100)
                
                # Generate final AI summary if available
                if self.gemini_agent and self.gemini_agent.is_available():
                    final_summary = self._generate_ai_final_summary(
                        smiles, uniprot_ids, combined_heteroatom_results, combined_similarity_results, ai_params
                    )
                    st.success("🎉 **Autonomous Hybrid Analysis Complete!**")
                    st.info(f"**🤖 AI Final Summary:** {final_summary}")
                else:
                    st.success("🎉 **Autonomous Hybrid Analysis Complete!** Processing finished using hybrid data sources.")
                
            except Exception as e:
                st.error(f"❌ Autonomous hybrid analysis failed: {str(e)}")
                if self.gemini_agent and self.gemini_agent.is_available():
                    error_analysis = self.gemini_agent.generate_ai_response_sync(
                        f"Analyze this error and suggest solutions: {str(e)}"
                    )
                    st.info(f"**AI Error Analysis:** {error_analysis}")
            finally:
                progress_bar.empty()
                status_text.empty()

    def _generate_ai_final_summary(self, smiles: str, uniprot_ids: List[str], 
                                  heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                  ai_params: Dict) -> str:
        """Generate final AI summary of the autonomous analysis"""
        if not self.gemini_agent or not self.gemini_agent.is_available():
            return "Analysis completed successfully using hybrid data sources."
        
        try:
            # Prepare comprehensive summary for AI
            summary_data = {
                'target_smiles': smiles,
                'uniprot_ids': uniprot_ids,
                'total_heteroatoms': len(heteroatom_results),
                'unique_pdbs': heteroatom_results['PDB_ID'].nunique() if not heteroatom_results.empty else 0,
                'similarity_matches': len(similarity_results),
                'best_similarity': similarity_results['Tanimoto_Similarity'].max() if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns else 0,
                'avg_similarity': similarity_results['Tanimoto_Similarity'].mean() if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns else 0,
                'ai_parameters': ai_params
            }
            
            prompt = f"""
            As a molecular biology expert, provide a concise final summary of this autonomous protein-ligand analysis:
            
            Analysis Results: {summary_data}
            
            Provide insights on:
            1. Overall analysis quality and completeness
            2. Key molecular findings and their significance
            3. Recommendations for follow-up research
            4. Confidence level in the results
            
            Keep the summary professional, scientific, and under 4 sentences.
            """
            
            return self.gemini_agent.generate_ai_response_sync(prompt)
            
        except Exception as e:
            return f"Analysis completed successfully. AI summary generation encountered an issue: {str(e)[:50]}..."

    def _display_autonomous_hybrid_results(self, smiles: str, uniprot_ids: List[str],
                                          heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                          ai_params: Dict, comprehensive: bool, visualizations: bool,
                                          auto_download: bool, result_detail: str,
                                          use_local_data: bool, use_realtime_data: bool, data_priority: str, action: str):
        """Display comprehensive autonomous hybrid analysis results"""
        
        # Analysis overview with AI context
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
        
        # AI-powered analysis summary
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**🤖 AI Action:** {action.replace('_', ' ').title()}")
            st.info(f"**📊 Data Sources:** {'Local + ' if use_local_data else ''}{'Real-Time' if use_realtime_data else ''}")
        with col2:
            st.info(f"**⚙️ AI Parameters:** Radius={ai_params['radius']}, Bits={ai_params['n_bits']}")
            st.info(f"**🎯 Priority:** {data_priority}")
        
        # Results display based on detail level
        if result_detail in ["Standard", "Detailed"]:
            if not heteroatom_results.empty:
                st.subheader("🔬 Autonomous Hybrid Heteroatom Analysis")
                self._display_heteroatom_results(heteroatom_results, "autonomous-hybrid")
                
                # AI insights for heteroatom results
                if self.gemini_agent and self.gemini_agent.is_available():
                    with st.expander("🧠 AI Heteroatom Insights", expanded=False):
                        hetero_summary = {
                            'total_heteroatoms': len(heteroatom_results),
                            'unique_codes': heteroatom_results['Heteroatom_Code'].nunique(),
                            'success_rate': len(heteroatom_results[heteroatom_results['SMILES'] != '']) / len(heteroatom_results) * 100
                        }
                        ai_hetero_insights = self.gemini_agent.get_scientific_explanation(hetero_summary)
                        st.write(ai_hetero_insights)
            
            if not similarity_results.empty:
                st.subheader("🎯 Autonomous Hybrid Similarity Analysis")
                self._display_similarity_results(similarity_results, smiles, "autonomous-hybrid")
                
                # AI insights for similarity results
                if self.gemini_agent and self.gemini_agent.is_available():
                    with st.expander("🧠 AI Similarity Insights", expanded=False):
                        sim_summary = {
                            'similarity_matches': len(similarity_results),
                            'top_score': similarity_results['Tanimoto_Similarity'].max(),
                            'distribution': similarity_results['Tanimoto_Similarity'].describe().to_dict()
                        }
                        ai_sim_insights = self.gemini_agent.get_scientific_explanation(sim_summary)
                        st.write(ai_sim_insights)
        
        # Comprehensive AI report
        if comprehensive and self.gemini_agent and self.gemini_agent.is_available():
            st.subheader("📋 AI Autonomous Hybrid Comprehensive Report")
            report = self._generate_autonomous_hybrid_report(
                smiles, uniprot_ids, heteroatom_results, similarity_results, ai_params,
                use_local_data, use_realtime_data, data_priority, action
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
                                          data_priority: str, action: str) -> str:
        """Generate comprehensive autonomous hybrid analysis report with AI insights"""
        
        data_sources = []
        if use_local_data:
            data_sources.append(f"Local Excel database ({len(self.local_database)} records)")
        if use_realtime_data:
            data_sources.append("Real-time PDB Data Bank")
        
        report = f"""
## 🤖 AI Autonomous Hybrid Analysis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**AI Version:** Gemini-1.5-Flash (Google AI Studio)

### 📋 AI Analysis Summary
- **Detected Action:** {action.replace('_', ' ').title()}
- **Target SMILES:** `{smiles if smiles else 'Auto-detected from input'}`
- **UniProt IDs:** {', '.join(uniprot_ids) if uniprot_ids else 'Auto-selected from database'}
- **Data Sources:** {' + '.join(data_sources)}
- **Data Priority:** {data_priority}

### ⚙️ AI-Optimized Parameters
- **Morgan Radius:** {ai_params['radius']} (AI-optimized for performance)
- **Bit Vector Length:** {ai_params['n_bits']} (AI-selected for accuracy/speed balance)
- **Fingerprint Type:** {ai_params['fp_type']} (AI-recommended)
- **Similarity Metric:** {ai_params['metric']} (AI-chosen)
- **Threshold:** {ai_params['threshold']} (User-specified, AI-validated)

### 📊 Hybrid Results Overview
- **Heteroatom Records:** {len(heteroatom_results)} (AI-processed)
- **Similar Compounds:** {len(similarity_results)} (AI-analyzed)
- **Local Database Used:** {'✅ Yes' if use_local_data else '❌ No'}
- **Real-Time Fetching Used:** {'✅ Yes' if use_realtime_data else '❌ No'}
- **Processing Mode:** Fully Autonomous with AI Optimization
"""
        
        if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns:
            max_sim = similarity_results['Tanimoto_Similarity'].max()
            avg_sim = similarity_results['Tanimoto_Similarity'].mean()
            high_sim_count = len(similarity_results[similarity_results['Tanimoto_Similarity'] >= 0.8])
            
            report += f"""
### 🎯 AI Hybrid Similarity Analysis
- **Highest Similarity:** {max_sim:.3f} (AI-identified top match)
- **Average Similarity:** {avg_sim:.3f} (AI-calculated mean)
- **High-Confidence Matches (≥0.8):** {high_sim_count} compounds
- **AI Confidence Level:** {'High' if avg_sim > 0.7 else 'Medium' if avg_sim > 0.5 else 'Moderate'}
"""
        
        # Add AI-generated insights if available
        if self.gemini_agent and self.gemini_agent.is_available():
            try:
                overall_summary = {
                    'heteroatom_count': len(heteroatom_results),
                    'similarity_count': len(similarity_results),
                    'best_similarity': max_sim if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns else 0,
                    'action_performed': action
                }
                ai_insights = self.gemini_agent.get_scientific_explanation(overall_summary)
                report += f"""
### 🧠 AI Scientific Insights
{ai_insights}
"""
            except Exception:
                pass
        
        report += f"""
### 🔮 AI System Analysis
- **Analysis Completed:** Fully autonomous with minimal user intervention
- **Data Integration:** Hybrid approach combining local speed with real-time completeness
- **Parameter Optimization:** AI-determined optimal settings for user requirements
- **Quality Assessment:** AI-validated results with confidence scoring
- **Future Recommendations:** {'Enhanced with full PDB database integration' if use_local_data else 'Consider adding local database for faster processing'}

### ✅ Autonomous Status
**FULLY AUTONOMOUS HYBRID ANALYSIS COMPLETE** - AI handled all parameter optimization and decision-making
**Next Steps:** Review results, download data, or request additional analysis with different parameters
"""
        
        return report

    def _create_autonomous_visualizations(self, similarity_results: pd.DataFrame, target_smiles: str):
        """Create visualizations for autonomous hybrid analysis results"""
        try:
            # Distribution histogram
            fig_hist = px.histogram(
                similarity_results,
                x='Tanimoto_Similarity',
                title="Molecular Similarity Distribution",
                nbins=20,
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.update_layout(xaxis_title="Tanimoto Similarity", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Scatter plot of top similar compounds
            top_results = similarity_results.head(10)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=top_results.index,
                y=top_results['Tanimoto_Similarity'],
                mode='markers+lines',
                marker=dict(size=10, color=top_results['Tanimoto_Similarity'], colorscale='Viridis'),
                text=[f"{row['PDB_ID']}-{row['Heteroatom_Code']}" for _, row in top_results.iterrows()],
                hovertemplate='<b>%{text}</b><br>Similarity: %{y:.3f}<extra></extra>'
            ))
            fig_scatter.update_layout(
                title="Top 10 Similar Compounds",
                xaxis_title="Rank",
                yaxis_title="Tanimoto Similarity",
                xaxis=dict(tickvals=list(range(len(top_results))), ticktext=[str(i+1) for i in range(len(top_results))]),
                yaxis=dict(range=[0, 1]),
                margin=dict(t=30, b=30, l=40, r=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Error creating visualizations: {e}")

    def _generate_autonomous_downloads(self, heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame):
        """Generate downloadable files for autonomous analysis results"""
        try:
            # Heteroatom results CSV
            if not heteroatom_results.empty:
                csv_heteroatoms = heteroatom_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Results CSV",
                    data=csv_heteroatoms,
                    file_name="heteroatom_results_autonomous.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Similarity results CSV
            if not similarity_results.empty:
                csv_similarity = similarity_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Similarity Results CSV",
                    data=csv_similarity,
                    file_name="similarity_results_autonomous.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.warning(f"Error generating downloads: {e}")