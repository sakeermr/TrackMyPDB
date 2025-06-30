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

def get_base64_image(image_path):
    """Convert image to base64 string for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return ""

# Page configuration
st.set_page_config(
    page_title="TrackMyPDB",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Apple-like design with pale green theme
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        background-color: #f8fffe;
    }
    .stApp {
        background-color: #f0f9f5;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    .metric-card {
        background: rgba(244, 255, 250, 0.8);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(129, 199, 132, 0.3);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .watermark {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(240, 249, 245, 0.95);
        padding: 12px 20px;
        border-radius: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        font-size: 0.85rem;
        color: #2E7D32;
        font-weight: 500;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
    .linkedin-link {
        color: #2E7D32;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.3s ease;
    }
    .linkedin-link:hover {
        color: #1B5E20;
        transform: scale(1.05);
    }
    .linkedin-icon {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(76, 175, 80, 0.3);
        text-align: center;
        color: #4CAF50;
        font-size: 0.9rem;
    }
    .sidebar .sidebar-content {
        background-color: #f4fffb;
    }
    .stSelectbox > div > div {
        background-color: #f4fffb;
    }
    .stTextInput > div > div > input {
        background-color: #f4fffb;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .stTextArea > div > div > textarea {
        background-color: #f4fffb;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def show_sidebar_watermark():
    """Display watermark at bottom of sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(244, 255, 250, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        text-align: center;
        font-size: 0.85rem;
        color: #2E7D32;
        backdrop-filter: blur(10px);
    ">
        <strong>Developed and released by Standard Seed Corporation</strong><br>
        <small style="color: #666;">TrackMyPDB V 1.0</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Add SSC logo if it exists
    if os.path.exists("ssc.png"):
        st.sidebar.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <img src="data:image/png;base64,{}" width="80" style="
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(76, 175, 80, 0.2);
                margin-top: 0.5rem;
            ">
        </div>
        """.format(get_base64_image("ssc.png")), unsafe_allow_html=True)
    else:
        st.sidebar.info("💡 Add ssc.png to display SSC logo")

def show_footer():
    """Display footer with license information"""
    st.markdown("""
    <div class="footer">
        <p>📄 Licensed under MIT License - Open Source Project</p>
        <p>🧬 TrackMyPDB - Bioinformatics Pipeline for Protein Structure Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.title("🧬 TrackMyPDB")
    st.markdown("### *Protein Structure Heteroatom Extraction & Molecular Similarity Analysis*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Home", "🔍 Heteroatom Extraction", "🧪 Similarity Analysis", "📊 Complete Pipeline"]
    )
    
    # Add watermark at bottom of sidebar
    show_sidebar_watermark()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Heteroatom Extraction":
        show_extraction_page()
    elif page == "🧪 Similarity Analysis":
        show_similarity_page()
    elif page == "📊 Complete Pipeline":
        show_complete_pipeline()
    
    # Show footer
    show_footer()

def show_home_page():
    """Display home page with project overview"""
    
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔬 Heteroatom Extraction Tool
        - **Purpose**: Extract ALL heteroatoms from PDB structures
        - **Input**: UniProt protein identifiers
        - **Output**: Comprehensive ligand database with SMILES
        - **Features**: Multi-source data fetching, robust error handling
        """)
        
        st.markdown("""
        #### 📊 Key Capabilities
        - ✅ **Comprehensive extraction**: Processes ALL heteroatoms
        - ✅ **Multi-source data**: RCSB PDB and PubChem APIs
        - ✅ **Progress tracking**: Real-time status updates
        - ✅ **Error handling**: Graceful API failure management
        """)
    
    with col2:
        st.markdown("""
        #### 🧪 Molecular Similarity Analyzer
        - **Purpose**: Find molecules similar to target compound
        - **Input**: Target SMILES structure
        - **Output**: Ranked similarity results
        - **Method**: Morgan fingerprints + Tanimoto similarity
        """)
        
        st.markdown("""
        #### 🎯 Analysis Features
        - ✅ **Morgan fingerprints**: Industry-standard representations
        - ✅ **Tanimoto similarity**: Robust similarity metrics
        - ✅ **Rich visualizations**: Interactive plots and charts
        - ✅ **Statistical reports**: Comprehensive analysis
        """)
    
    # Workflow diagram
    st.markdown('<div class="section-header">🔄 Workflow</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ```
    UniProt IDs → PDB Structures → Heteroatom Extraction → SMILES Database
                                                              ↓
    Target SMILES → Fingerprint Computation → Similarity Analysis → Results CSV
    ```
    """)
    
    # Quick start
    st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Navigate** to "🔍 Heteroatom Extraction"
    2. **Enter** your UniProt IDs (e.g., Q9UNQ0, P37231, P06276)
    3. **Run** extraction to build heteroatom database
    4. **Switch** to "🧪 Similarity Analysis"
    5. **Input** your target SMILES structure
    6. **Analyze** molecular similarities
    7. **Download** results as CSV
    """)

def show_extraction_page():
    """Display heteroatom extraction interface"""
    
    st.markdown('<div class="section-header">🔍 Heteroatom Extraction</div>', unsafe_allow_html=True)
    
    # Input section
    st.subheader("📋 Input Parameters")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # UniProt IDs input
        uniprot_input = st.text_area(
            "UniProt IDs",
            placeholder="Enter UniProt IDs (one per line or comma-separated)\nExample: Q9UNQ0, P37231, P06276",
            height=100,
            help="Enter protein UniProt identifiers to extract heteroatoms from associated PDB structures"
        )
        
        # Parse UniProt IDs
        if uniprot_input:
            # Handle both comma-separated and line-separated input
            uniprot_ids = []
            for line in uniprot_input.strip().split('\n'):
                for up_id in line.split(','):
                    up_id = up_id.strip()
                    if up_id:
                        uniprot_ids.append(up_id)
            
            st.info(f"Found {len(uniprot_ids)} UniProt IDs: {', '.join(uniprot_ids)}")
    
    with col2:
        st.markdown("#### 📊 Extraction Settings")
        
        # Download existing results
        if os.path.exists("heteroatom_results.csv"):
            st.success("Previous results found!")
            if st.button("📥 Load Previous Results"):
                df = pd.read_csv("heteroatom_results.csv")
                st.session_state['heteroatom_data'] = df
                st.success("Previous results loaded!")
        
        # Clear results
        if st.button("🗑️ Clear Results"):
            if 'heteroatom_data' in st.session_state:
                del st.session_state['heteroatom_data']
            if os.path.exists("heteroatom_results.csv"):
                os.remove("heteroatom_results.csv")
            st.success("Results cleared!")
    
    # Run extraction
    if st.button("🚀 Start Heteroatom Extraction", type="primary"):
        if not uniprot_input:
            st.error("Please enter at least one UniProt ID")
            return
        
        # Initialize extractor
        extractor = HeteroatomExtractor()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Run extraction
            with st.spinner("Extracting heteroatoms..."):
                df = extractor.extract_heteroatoms(uniprot_ids, progress_callback=update_progress)
            
            # Store results
            st.session_state['heteroatom_data'] = df
            
            # Save to CSV
            df.to_csv("heteroatom_results.csv", index=False)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("Heteroatom extraction completed successfully!")
            
        except Exception as e:
            st.error(f"Error during extraction: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Display results
    if 'heteroatom_data' in st.session_state:
        df = st.session_state['heteroatom_data']
        
        st.markdown('<div class="section-header">📊 Extraction Results</div>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("PDB Structures", df['PDB_ID'].nunique())
        with col3:
            st.metric("Unique Heteroatoms", df['Heteroatom_Code'].nunique())
        with col4:
            st.metric("Records with SMILES", len(df[df['SMILES'] != '']))
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv,
            file_name=f"heteroatom_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_similarity_page():
    """Display molecular similarity analysis interface"""
    
    st.markdown('<div class="section-header">🧪 Molecular Similarity Analysis</div>', unsafe_allow_html=True)
    
    # Check if heteroatom data exists
    if 'heteroatom_data' not in st.session_state:
        if os.path.exists("heteroatom_results.csv"):
            df = pd.read_csv("heteroatom_results.csv")
            st.session_state['heteroatom_data'] = df
            st.success("Loaded previous heteroatom extraction results!")
        else:
            st.warning("No heteroatom data found. Please run heteroatom extraction first.")
            return
    
    # Input section
    st.subheader("🎯 Target Molecule")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_smiles = st.text_input(
            "Target SMILES Structure",
            placeholder="Enter SMILES string (e.g., CCO for ethanol)",
            help="Enter the SMILES representation of your target molecule"
        )
        
        # SMILES validation
        if target_smiles:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(target_smiles)
                if mol is not None:
                    st.success("✅ Valid SMILES structure")
                else:
                    st.error("❌ Invalid SMILES structure")
            except:
                st.error("❌ Error validating SMILES")
    
    with col2:
        st.markdown("#### ⚙️ Analysis Parameters")
        
        top_n = st.slider("Number of Results", 10, 100, 50)
        min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.2, 0.1)
        
        st.markdown("#### 📊 Fingerprint Settings")
        radius = st.selectbox("Morgan Radius", [1, 2, 3], index=1)
        n_bits = st.selectbox("Fingerprint Bits", [1024, 2048, 4096], index=1)
    
    # Run analysis
    if st.button("🔍 Analyze Molecular Similarity", type="primary"):
        if not target_smiles:
            st.error("Please enter a target SMILES structure")
            return
        
        # Initialize analyzer
        analyzer = MolecularSimilarityAnalyzer(radius=radius, n_bits=n_bits)
        
        try:
            # Run analysis
            with st.spinner("Analyzing molecular similarity..."):
                heteroatom_df = st.session_state['heteroatom_data']
                similarity_results = analyzer.analyze_similarity(
                    target_smiles=target_smiles,
                    heteroatom_df=heteroatom_df,
                    top_n=top_n,
                    min_similarity=min_similarity
                )
            
            # Store results
            st.session_state['similarity_results'] = similarity_results
            
            # Enhanced download functionality
            if not similarity_results.empty:
                # Prepare the CSV with exactly the columns requested: 
                # PDB_ID, Heteroatom_Code, Chemical_Name, SMILES, Tanimoto_Similarity, Formula
                download_df = similarity_results[[
                    'PDB_ID', 
                    'Heteroatom_Code', 
                    'Chemical_Name', 
                    'SMILES', 
                    'Tanimoto_Similarity', 
                    'Formula'
                ]].copy()
                
                # Format similarity scores
                download_df['Tanimoto_Similarity'] = download_df['Tanimoto_Similarity'].round(4)
                
                # Sort by best Tanimoto scores (highest first)
                download_df = download_df.sort_values('Tanimoto_Similarity', ascending=False).reset_index(drop=True)
                
                # Save to CSV
                download_df.to_csv("similarity_results.csv", index=False)
                
                # Enhanced download section
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📊 Total Results", len(download_df))
                    st.metric("🏆 Best Score", f"{download_df['Tanimoto_Similarity'].max():.4f}")
                
                with col2:
                    # Download button with enhanced CSV
                    csv_data = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Similarity Results (CSV)",
                        data=csv_data,
                        file_name=f"TrackMyPDB_similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Downloads the complete similarity analysis results sorted by best Tanimoto scores"
                    )
                
                # Show preview of downloadable data
                st.subheader("📋 Download Preview (Top 10 Results)")
                st.dataframe(
                    download_df.head(10),
                    use_container_width=True,
                    column_config={
                        "PDB_ID": "PDB ID",
                        "Heteroatom_Code": "Ligand Code",
                        "Chemical_Name": "Chemical Name", 
                        "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
                        "Tanimoto_Similarity": st.column_config.NumberColumn(
                            "Similarity Score",
                            help="Higher scores indicate better matches",
                            format="%.4f"
                        ),
                        "Formula": "Molecular Formula"
                    },
                    hide_index=True
                )
            
            else:
                st.warning("⚠️ No results found above the similarity threshold. Try lowering the minimum similarity value.")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

def show_complete_pipeline():
    """Display complete pipeline interface"""
    
    st.markdown('<div class="section-header">📊 Complete Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Run the complete TrackMyPDB pipeline in one go:
    1. Extract heteroatoms from UniProt proteins
    2. Analyze molecular similarity to target compound
    3. Generate comprehensive results
    """)
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 UniProt Input")
        uniprot_input = st.text_area(
            "UniProt IDs",
            placeholder="Q9UNQ0, P37231, P06276",
            height=100
        )
    
    with col2:
        st.subheader("🎯 Target Molecule")
        target_smiles = st.text_input(
            "Target SMILES",
            placeholder="CCO"
        )
    
    # Parameters
    st.subheader("⚙️ Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Top Results", 10, 100, 50)
    with col2:
        min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.2, 0.1)
    with col3:
        radius = st.selectbox("Morgan Radius", [1, 2, 3], index=1)
    
    # Run complete pipeline
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        if not uniprot_input or not target_smiles:
            st.error("Please provide both UniProt IDs and target SMILES")
            return
        
        # Parse UniProt IDs
        uniprot_ids = []
        for line in uniprot_input.strip().split('\n'):
            for up_id in line.split(','):
                up_id = up_id.strip()
                if up_id:
                    uniprot_ids.append(up_id)
        
        # Step 1: Heteroatom Extraction
        st.info("Step 1: Extracting heteroatoms...")
        extractor = HeteroatomExtractor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress * 0.7)  # Use 70% for extraction
            status_text.text(f"Extraction: {message}")
        
        try:
            heteroatom_df = extractor.extract_heteroatoms(uniprot_ids, progress_callback=update_progress)
            
            # Step 2: Similarity Analysis
            status_text.text("Step 2: Analyzing molecular similarity...")
            progress_bar.progress(0.7)
            
            analyzer = MolecularSimilarityAnalyzer(radius=radius)
            similarity_results = analyzer.analyze_similarity(
                target_smiles=target_smiles,
                heteroatom_df=heteroatom_df,
                top_n=top_n,
                min_similarity=min_similarity
            )
            
            progress_bar.progress(1.0)
            status_text.text("Pipeline completed successfully!")
            
            # Save results
            heteroatom_df.to_csv("complete_pipeline_heteroatoms.csv", index=False)
            if not similarity_results.empty:
                similarity_results.to_csv("complete_pipeline_similarity.csv", index=False)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv1 = heteroatom_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Results",
                    data=csv1,
                    file_name=f"heteroatoms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if not similarity_results.empty:
                    csv2 = similarity_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Similarity Results",
                        data=csv2,
                        file_name=f"similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main() 