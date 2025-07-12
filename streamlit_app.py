"""
TrackMyPDB - Restructured Application
@author Anu Gamage

A comprehensive bioinformatics pipeline with 3 distinct modes:
1. Manual Mode - Real-time PDB fetching with manual inputs
2. AI-Powered Mode - Interactive chatbot with step-by-step confirmation
3. Fully Autonomous Mode - AI works autonomously without user intervention

Licensed under MIT License - Open Source Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
from backend.heteroatom_extractor import HeteroatomExtractor
from backend.agent_core import TrackMyPDBAgent
from backend.nl_interface import NaturalLanguageInterface

try:
    from backend.similarity_analyzer import SimilarityAnalyzer
    RDKIT_AVAILABLE = True
except ImportError:
    from backend.similarity_analyzer_simple import SimilarityAnalyzer
    RDKIT_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="TrackMyPDB - Molecular Analysis Platform",
    page_icon="🧬",
    layout="wide"
)

# Load local PDB database from Excel files
@st.cache_data
def load_local_pdb_database():
    """Load and combine all local PDB data files from the data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    combined_data = pd.DataFrame()
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(data_dir, file)
                    df = pd.read_csv(file_path)
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
                    st.sidebar.success(f"✅ Loaded {file}")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to load {file}: {e}")
    else:
        st.sidebar.warning("⚠️ Data directory not found. Using empty database.")
    
    return combined_data

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = "Manual Mode"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = TrackMyPDBAgent()

if "heteroatom_extractor" not in st.session_state:
    st.session_state.heteroatom_extractor = HeteroatomExtractor()

if "local_database" not in st.session_state:
    st.session_state.local_database = load_local_pdb_database()

if "similarity_analyzer" not in st.session_state:
    st.session_state.similarity_analyzer = SimilarityAnalyzer()

if "nl_interface" not in st.session_state:
    st.session_state.nl_interface = NaturalLanguageInterface(
        agent=st.session_state.agent,
        local_database=st.session_state.local_database
    )

def display_database_status():
    """Display status of local PDB database"""
    with st.expander("📊 Database Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Local Database Size", len(st.session_state.local_database))
        
        with col2:
            data_source = "Excel files (PDB-derived)" if not st.session_state.local_database.empty else "Empty"
            st.metric("Data Source", data_source)
        
        with col3:
            rdkit_status = "✅ Available" if RDKIT_AVAILABLE else "❌ Not Available"
            st.metric("RDKit Status", rdkit_status)
        
        if not st.session_state.local_database.empty:
            st.write("**Database Preview:**")
            st.dataframe(st.session_state.local_database.head(3), use_container_width=True)

def main():
    """Main application interface with three distinct modes"""
    st.title("🧬 TrackMyPDB - Molecular Analysis Platform")
    
    # Display database status
    display_database_status()
    
    # Mode selection in sidebar
    st.sidebar.title("🎯 Analysis Mode Selection")
    
    mode = st.sidebar.selectbox(
        "Choose your analysis mode:",
        ["Manual Mode", "AI-Powered Mode", "Fully Autonomous Mode"],
        help=""":
        **Manual Mode**: Provide UniProt IDs manually, real-time PDB fetching
        **AI-Powered Mode**: Interactive chatbot with step-by-step confirmation  
        **Fully Autonomous Mode**: AI works autonomously with advanced predictions
        """
    )
    
    st.session_state.mode = mode
    
    # Mode-specific information
    st.sidebar.markdown("---")
    if mode == "Manual Mode":
        st.sidebar.markdown("""
        **Manual Mode Features:**
        - 🔗 Real-time PDB Data Bank fetching
        - ⚙️ Manual parameter configuration
        - 📊 Three analysis stages available
        - 🎛️ Full user control
        """)
    elif mode == "AI-Powered Mode":
        st.sidebar.markdown("""
        **AI-Powered Mode Features:**
        - 🤖 Interactive chatbot interface
        - ❓ Step-by-step questions
        - 📁 Uses local Excel database
        - ✅ User confirmation required
        """)
    else:  # Fully Autonomous Mode
        st.sidebar.markdown("""
        **Fully Autonomous Mode Features:**
        - 🚀 Complete AI automation
        - 📁 Uses local Excel database
        - 🔄 Continuous processing
        - 📋 Comprehensive reports
        """)
    
    # Route to appropriate interface using NaturalLanguageInterface
    if mode == "Manual Mode":
        st.session_state.nl_interface.render_manual_mode_interface()
    elif mode == "AI-Powered Mode":
        st.session_state.nl_interface.render_ai_powered_mode_interface()
    else:  # Fully Autonomous Mode
        st.session_state.nl_interface.render_fully_autonomous_mode_interface()

# Footer and additional information
def render_footer():
    """Render application footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📝 Current Mode Behavior:**
        - **Manual**: Real-time PDB fetching + manual inputs
        - **AI-Powered**: Excel database + guided questions  
        - **Autonomous**: Excel database + full automation
        """)
    
    with col2:
        st.markdown("""
        **🔮 Future Enhancements:**
        - Live PDB connection for AI modes
        - Enhanced parameter optimization
        - Advanced visualization features
        """)
    
    with col3:
        st.markdown("""
        **📊 Data Sources:**
        - Manual Mode: PDB Data Bank (real-time)
        - AI Modes: Local Excel files (PDB-derived)
        - Future: Unified live PDB access
        """)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>TrackMyPDB v2.0 - Three-Mode Molecular Analysis Platform</p>
        <p>Developed by Anu Gamage | MIT License | Open Source Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    render_footer()