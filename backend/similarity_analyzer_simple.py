"""
TrackMyPDB Molecular Similarity Analyzer (Simplified Version)
@author Anu Gamage

This is a simplified version that can work without RDKit for basic testing.
Licensed under MIT License - Open Source Project
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings

warnings.filterwarnings('ignore')


class MolecularSimilarityAnalyzer:
    """Simplified molecular similarity analyzer for testing purposes"""

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.fingerprints = {}
        self.valid_molecules = {}

    def smiles_to_fingerprint(self, smiles):
        """Simplified fingerprint generation (placeholder)"""
        if not smiles or pd.isna(smiles) or smiles.strip() == '':
            return None
        
        # Placeholder: Simple hash-based fingerprint for testing
        # Use a reasonable modulo to prevent integer overflow
        hash_value = abs(hash(smiles.strip())) % 100000  # Use smaller modulo to prevent overflow
        return hash_value

    def calculate_tanimoto_similarity(self, fp1, fp2):
        """Simplified Tanimoto similarity calculation"""
        if fp1 is None or fp2 is None:
            return 0.0
        
        # Placeholder: Simple similarity based on hash difference
        # In real implementation, this would use RDKit's Tanimoto similarity
        try:
            max_val = max(fp1, fp2)
            min_val = min(fp1, fp2)
            if max_val == 0:
                return 0.0
            # Ensure the result is a float and within valid range
            similarity = float(min_val) / float(max_val)
            return min(max(similarity, 0.0), 1.0)  # Clamp between 0 and 1
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0.0

    def load_and_process_dataframe(self, df, smiles_column='SMILES'):
        """Load DataFrame and process SMILES to fingerprints"""
        st.info("Processing DataFrame and computing fingerprints...")
        st.warning("⚠️ Using simplified fingerprints - Install RDKit for full functionality")

        valid_df = df[df[smiles_column].notna() & (df[smiles_column] != '')].copy()
        st.info(f"Found {len(valid_df)} entries with SMILES out of {len(df)} total")

        fingerprints = []
        valid_indices = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (idx, row) in enumerate(valid_df.iterrows()):
            progress = (i + 1) / len(valid_df)
            progress_bar.progress(progress)
            status_text.text(f"Computing fingerprints... {i+1}/{len(valid_df)}")
            
            smiles = row[smiles_column]
            fp = self.smiles_to_fingerprint(smiles)

            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)

        progress_bar.empty()
        status_text.empty()

        processed_df = valid_df.loc[valid_indices].copy()
        processed_df['Fingerprint'] = fingerprints

        st.success(f"Successfully processed {len(processed_df)} molecules with simplified fingerprints")
        return processed_df

    def find_similar_ligands(self, target_smiles, processed_df, top_n=50, min_similarity=0.0):
        """Find ligands similar to target molecule"""
        st.info(f"Analyzing similarity to target: {target_smiles}")
        st.warning("⚠️ Using simplified similarity - Install RDKit for accurate results")

        target_fp = self.smiles_to_fingerprint(target_smiles)
        if target_fp is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        similarities = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (idx, row) in enumerate(processed_df.iterrows()):
            progress = (i + 1) / len(processed_df)
            progress_bar.progress(progress)
            status_text.text(f"Computing similarities... {i+1}/{len(processed_df)}")
            
            ligand_fp = row['Fingerprint']  
            similarity = self.calculate_tanimoto_similarity(target_fp, ligand_fp)
            similarities.append(similarity)

        progress_bar.empty()
        status_text.empty()

        result_df = processed_df.copy()
        result_df['Tanimoto_Similarity'] = similarities

        result_df = result_df[result_df['Tanimoto_Similarity'] >= min_similarity]
        result_df = result_df.sort_values('Tanimoto_Similarity', ascending=False)

        top_results = result_df.head(top_n)

        st.success(f"Found {len(result_df)} ligands above similarity threshold {min_similarity}")
        st.info(f"Returning top {len(top_results)} results")

        return top_results

    def create_similarity_report(self, target_smiles, similarity_results):
        """Create comprehensive similarity analysis report"""
        st.subheader("🎯 Molecular Similarity Analysis Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Target SMILES", target_smiles)
            st.metric("Total Similar Ligands", len(similarity_results))
        
        with col2:
            if len(similarity_results) > 0:
                st.metric("Highest Similarity", f"{similarity_results['Tanimoto_Similarity'].max():.4f}")
                st.metric("Average Similarity", f"{similarity_results['Tanimoto_Similarity'].mean():.4f}")

        if len(similarity_results) > 0:
            st.subheader("🔝 Top 10 Most Similar Ligands")
            top_10 = similarity_results.head(10)
            
            display_df = top_10[['PDB_ID', 'Heteroatom_Code', 'Chemical_Name', 'Tanimoto_Similarity', 'SMILES']].copy()
            display_df['Tanimoto_Similarity'] = display_df['Tanimoto_Similarity'].round(4)
            st.dataframe(display_df, use_container_width=True)

    def create_interactive_plots(self, similarity_results, target_smiles):
        """Create interactive visualizations using Plotly"""
        if len(similarity_results) == 0:
            st.warning("No data to plot")
            return

        # Simple histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=similarity_results['Tanimoto_Similarity'], 
                                  nbinsx=20, name='Similarity Distribution'))
        fig.update_layout(title=f"Similarity Distribution for {target_smiles[:30]}...",
                         xaxis_title='Tanimoto Similarity',
                         yaxis_title='Frequency')
        
        st.plotly_chart(fig, use_container_width=True)

    def analyze_similarity(self, target_smiles, heteroatom_df, top_n=50, min_similarity=0.2):
        """Complete similarity analysis workflow"""
        processed_df = self.load_and_process_dataframe(heteroatom_df)

        if len(processed_df) == 0:
            st.error("No valid SMILES found in the DataFrame!")
            return pd.DataFrame()

        similarity_results = self.find_similar_ligands(
            target_smiles=target_smiles,
            processed_df=processed_df,
            top_n=top_n,
            min_similarity=min_similarity
        )

        self.create_similarity_report(target_smiles, similarity_results)

        st.subheader("📊 Similarity Visualizations")
        self.create_interactive_plots(similarity_results, target_smiles)

        return similarity_results 