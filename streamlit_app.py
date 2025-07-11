"""
TrackMyPDB - Streamlit Application
@author Anu Gamage

A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures
and finding molecularly similar compounds using fingerprint-based similarity analysis.

Licensed under MIT License - Open Source Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import base64
from datetime import datetime
from backend.agent_core import TrackMyPDBAgent
from backend.nl_interface import NaturalLanguageInterface

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from backend.heteroatom_extractor import HeteroatomExtractor
    # Try to import the full RDKit version first
    try:
        from backend.similarity_analyzer import MolecularSimilarityAnalyzer
        RDKIT_AVAILABLE = True
    except ImportError:
        # Fall back to simplified version
        from backend.similarity_analyzer_simple import MolecularSimilarityAnalyzer
        RDKIT_AVAILABLE = False
        st.warning("⚠️ RDKit not available - using simplified molecular similarity")
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TrackMyPDB - AI-Powered Protein-Ligand Analysis",
    page_icon="🧬",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_agent():
    agent = TrackMyPDBAgent()
    return agent

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()

if "nl_interface" not in st.session_state:
    st.session_state.nl_interface = NaturalLanguageInterface(st.session_state.agent)

def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return ""

def main():
    st.title("🧬 TrackMyPDB - AI-Powered Analysis")
    
    # Sidebar with analysis mode selection
    st.sidebar.title("Analysis Mode")
    mode = st.sidebar.selectbox(
        "Choose your interaction mode:",
        ["🤖 Natural Language", "📊 Traditional Interface"]
    )
    
    if mode == "🤖 Natural Language":
        st.session_state.nl_interface.render_chat_interface()
    else:
        render_traditional_interface()

def render_traditional_interface():
    """Render the traditional interface with AI enhancements"""
    st.subheader("Traditional Analysis Interface")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        ["🔍 Heteroatom Extraction", "🧪 Similarity Analysis", "📊 Complete Pipeline"]
    )
    
    if analysis_type == "🔍 Heteroatom Extraction":
        st.subheader("Heteroatom Extraction")
        
        uniprot_input = st.text_area(
            "UniProt IDs (one per line or comma-separated)",
            help="💡 AI Tip: Enter one or more UniProt IDs to analyze"
        )
        
        if st.button("🚀 Extract Heteroatoms"):
            if uniprot_input:
                uniprot_ids = [id.strip() for id in uniprot_input.replace(',', '\n').split('\n') if id.strip()]
                
                with st.spinner("🧬 Analyzing proteins..."):
                    results = st.session_state.agent.execute_action(
                        "extract_heteroatoms",
                        {"uniprot_ids": uniprot_ids}
                    )
                    
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success("✅ Analysis complete!")
                    st.session_state.last_heteroatom_results = results["results"]
                    st.write(results["results"])
    
    elif analysis_type == "🧪 Similarity Analysis":
        st.subheader("Similarity Analysis")
        
        if not hasattr(st.session_state, 'last_heteroatom_results'):
            st.warning("⚠️ Please run heteroatom extraction first")
            return
        
        smiles_input = st.text_area(
            "SMILES String",
            help="💡 AI Tip: Enter a SMILES string to find similar compounds"
        )
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="💡 AI Tip: Higher values mean more similar compounds"
        )
        
        if st.button("🔍 Analyze Similarity"):
            if smiles_input:
                with st.spinner("🧪 Analyzing similarity..."):
                    results = st.session_state.agent.execute_action(
                        "analyze_similarity",
                        {
                            "smiles": smiles_input,
                            "threshold": threshold
                        }
                    )
                    
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success("✅ Analysis complete!")
                    st.write(results["results"])
    
    else:  # Complete Pipeline
        st.subheader("Complete Pipeline Analysis")
        
        uniprot_input = st.text_area(
            "UniProt IDs (one per line or comma-separated)",
            help="💡 AI Tip: Enter one or more UniProt IDs to analyze"
        )
        
        smiles_input = st.text_area(
            "SMILES String",
            help="💡 AI Tip: Enter a SMILES string to find similar compounds"
        )
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="💡 AI Tip: Higher values mean more similar compounds"
        )
        
        if st.button("🚀 Run Complete Analysis"):
            if uniprot_input and smiles_input:
                uniprot_ids = [id.strip() for id in uniprot_input.replace(',', '\n').split('\n') if id.strip()]
                
                with st.spinner("🔄 Running complete analysis pipeline..."):
                    results = st.session_state.agent.execute_action(
                        "complete_pipeline",
                        {
                            "uniprot_ids": uniprot_ids,
                            "smiles": smiles_input,
                            "threshold": threshold
                        }
                    )
                    
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success("✅ Complete analysis finished!")
                    
                    st.subheader("📊 Heteroatom Results")
                    st.write(results["heteroatom_results"])
                    
                    st.subheader("🧪 Similarity Results")
                    st.write(results["similarity_results"])

if __name__ == "__main__":
    main()