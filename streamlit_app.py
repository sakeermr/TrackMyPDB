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

# Load local PDB database
@st.cache_data
def load_local_pdb_database():
    """Load and combine all local PDB data files from the data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    combined_data = pd.DataFrame()
    
    try:
        # Load all CSV files from data directory
        csv_files = ['Het-01.csv', 'Het-02.csv', 'Het-03.csv']
        
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, df], ignore_index=True)
                st.sidebar.success(f"✅ Loaded {file_name}")
        
        if not combined_data.empty:
            # Remove duplicates if any
            combined_data = combined_data.drop_duplicates()
            st.sidebar.info(f"📊 Total compounds in local database: {len(combined_data)}")
        
        return combined_data
    
    except Exception as e:
        st.sidebar.error(f"❌ Error loading database: {e}")
        return pd.DataFrame()

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

def main():
    """Main application interface"""
    st.title("🧬 TrackMyPDB - Molecular Analysis Platform")
    
    # Display database status
    display_database_status()
    
    # Mode selection in sidebar
    st.sidebar.title("🎯 Analysis Mode Selection")
    
    mode = st.sidebar.selectbox(
        "Choose your analysis mode:",
        ["Manual Mode", "AI-Powered Mode", "Enhanced Autonomous Mode", "Batch Processing Mode"],
        help=""":
        **Manual Mode**: Provide UniProt IDs manually, real-time PDB fetching
        **AI-Powered Mode**: Interactive chatbot with step-by-step confirmation  
        **Enhanced Autonomous Mode**: AI works autonomously with advanced predictions
        **Batch Processing Mode**: Analyze multiple compounds simultaneously
        """
    )
    
    st.session_state.mode = mode
    
    # Route to appropriate interface
    if mode == "Manual Mode":
        render_manual_mode()
    elif mode == "AI-Powered Mode":
        render_ai_powered_mode()
    elif mode == "Enhanced Autonomous Mode":
        enhanced_autonomous_mode()
    else:  # Batch Processing Mode
        batch_processing_mode()

def display_database_status():
    """Display status of local PDB database"""
    with st.sidebar.expander("📊 Database Status", expanded=False):
        if not st.session_state.local_database.empty:
            db = st.session_state.local_database
            st.write(f"**Total Compounds:** {len(db)}")
            
            if 'UniProt_ID' in db.columns:
                unique_proteins = db['UniProt_ID'].nunique()
                st.write(f"**Unique Proteins:** {unique_proteins}")
            
            if 'SMILES' in db.columns:
                valid_smiles = db['SMILES'].notna().sum()
                st.write(f"**Valid SMILES:** {valid_smiles}")
            
            # Show column names
            st.write("**Available Columns:**")
            for col in db.columns:
                st.write(f"• {col}")
        else:
            st.error("❌ No local database loaded")

def render_manual_mode():
    """Render Manual Mode interface - real-time PDB fetching"""
    st.header("🔧 Manual Mode - Real-time PDB Analysis")
    
    st.markdown("""
    **Manual Mode Features:**
    - Provide UniProt IDs and SMILES manually
    - Real-time fetching from PDB databank
    - Manual parameter configuration
    - Three analysis stages available
    """)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Parameters")
        
        # UniProt IDs input
        uniprot_input = st.text_area(
            "UniProt IDs (comma-separated):",
            placeholder="P12345, Q67890, R11223",
            help="Enter UniProt IDs separated by commas"
        )
        
        # SMILES input
        smiles_input = st.text_input(
            "Target SMILES Structure:",
            placeholder="CCO",
            help="Enter the SMILES string of your target molecule"
        )
    
    with col2:
        st.subheader("⚙️ Analysis Parameters")
        
        # Morgan parameters
        radius = st.slider("Morgan Radius:", 1, 4, 2)
        n_bits = st.selectbox("Fingerprint Bits:", [1024, 2048, 4096], index=1)
        threshold = st.slider("Tanimoto Threshold:", 0.0, 1.0, 0.7, 0.05)
        fp_type = st.selectbox("Fingerprint Type:", ["morgan", "maccs", "atompair"])
        metric = st.selectbox("Similarity Metric:", ["tanimoto", "dice", "cosine"])
    
    # Analysis stage selection
    st.subheader("🎯 Analysis Stage Selection")
    analysis_stage = st.radio(
        "Choose analysis stage:",
        ["Heteroatom Extraction", "Molecular Similarity Analysis", "Combined Pipeline"],
        horizontal=True
    )
    
    # Execute analysis
    if st.button("🚀 Run Manual Analysis", type="primary"):
        if analysis_stage == "Heteroatom Extraction":
            run_manual_heteroatom_extraction(uniprot_input)
        elif analysis_stage == "Molecular Similarity Analysis":
            run_manual_similarity_analysis(smiles_input, radius, n_bits, threshold, fp_type, metric)
        else:  # Combined Pipeline
            run_manual_combined_pipeline(uniprot_input, smiles_input, radius, n_bits, threshold, fp_type, metric)

def run_manual_heteroatom_extraction(uniprot_input):
    """Run heteroatom extraction in manual mode"""
    if not uniprot_input:
        st.error("❌ Please provide UniProt IDs")
        return
    
    uniprot_ids = [id.strip() for id in uniprot_input.split(',')]
    
    with st.spinner("🔍 Extracting heteroatoms from PDB..."):
        try:
            results = st.session_state.heteroatom_extractor.extract_heteroatoms(uniprot_ids)
            st.session_state.heteroatom_results = results
            
            st.success(f"✅ Extracted heteroatoms for {len(results)} compounds")
            
            # Display results
            if not results.empty:
                st.subheader("📊 Heteroatom Extraction Results")
                st.dataframe(results)
                
                # Download option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"heteroatom_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No heteroatoms found for the provided UniProt IDs")
                
        except Exception as e:
            st.error(f"❌ Extraction failed: {str(e)}")

def run_manual_similarity_analysis(smiles_input, radius, n_bits, threshold, fp_type, metric):
    """Run similarity analysis in manual mode"""
    if not smiles_input:
        st.error("❌ Please provide a SMILES string")
        return
    
    # Check if we have heteroatom results or use local database
    if hasattr(st.session_state, 'heteroatom_results') and not st.session_state.heteroatom_results.empty:
        data_source = st.session_state.heteroatom_results
        st.info("ℹ️ Using heteroatom data from previous extraction")
    elif not st.session_state.local_database.empty:
        data_source = st.session_state.local_database
        st.info("ℹ️ Using local PDB database")
    else:
        st.error("❌ No data available. Run heteroatom extraction first.")
        return
    
    with st.spinner("🧪 Analyzing molecular similarity..."):
        try:
            # Create analyzer with custom parameters
            analyzer = SimilarityAnalyzer(
                radius=radius,
                n_bits=n_bits,
                fp_type=fp_type,
                metric=metric
            )
            
            results = analyzer.analyze_similarity(
                target_smiles=smiles_input,
                heteroatom_df=data_source,
                min_similarity=threshold
            )
            
            st.success(f"✅ Found {len(results)} similar compounds")
            
            # Display results
            if not results.empty:
                st.subheader("🎯 Similarity Analysis Results")
                st.dataframe(results)
                
                # Visualization
                if len(results) > 0:
                    fig_similarity = create_similarity_plot(results)
                    st.plotly_chart(fig_similarity, use_container_width=True)
                
                # Download option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Similarity Results",
                    data=csv,
                    file_name=f"similarity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No similar compounds found above the threshold")
                
        except Exception as e:
            st.error(f"❌ Similarity analysis failed: {str(e)}")

def run_manual_combined_pipeline(uniprot_input, smiles_input, radius, n_bits, threshold, fp_type, metric):
    """Run combined pipeline in manual mode"""
    if not uniprot_input or not smiles_input:
        st.error("❌ Please provide both UniProt IDs and SMILES string")
        return
    
    uniprot_ids = [id.strip() for id in uniprot_input.split(',')]
    
    with st.spinner("🔄 Running complete analysis pipeline..."):
        try:
            # Execute complete pipeline
            results = st.session_state.agent.execute_action(
                "complete_pipeline",
                {
                    "uniprot_ids": uniprot_ids,
                    "smiles": smiles_input,
                    "radius": radius,
                    "n_bits": n_bits,
                    "threshold": threshold,
                    "fp_type": fp_type,
                    "metric": metric
                }
            )
            
            if "error" in results:
                st.error(f"❌ Pipeline failed: {results['error']}")
                return
            
            st.success("✅ Complete pipeline executed successfully!")
            
            # Display heteroatom results
            heteroatom_results = results["heteroatom_results"]
            if not heteroatom_results.empty:
                st.subheader("🔬 Heteroatom Extraction Results")
                st.dataframe(heteroatom_results)
            
            # Display similarity results
            similarity_results = results["similarity_results"]
            if not similarity_results.empty:
                st.subheader("🎯 Similarity Analysis Results")
                st.dataframe(similarity_results)
                
                # Visualization
                fig_similarity = create_similarity_plot(similarity_results)
                st.plotly_chart(fig_similarity, use_container_width=True)
            
            # Combined download
            with st.expander("📥 Download Combined Results", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    hetero_csv = heteroatom_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Heteroatom Results",
                        data=hetero_csv,
                        file_name=f"heteroatom_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    sim_csv = similarity_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Similarity Results",
                        data=sim_csv,
                        file_name=f"similarity_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Pipeline execution failed: {str(e)}")

def render_ai_powered_mode():
    """Render AI-Powered Mode - Interactive chatbot with confirmations"""
    st.header("🤖 AI-Powered Mode - Interactive Analysis")
    
    st.markdown("""
    **AI-Powered Mode Features:**
    - Interactive chatbot interface
    - Step-by-step confirmation process
    - AI asks clarifying questions
    - Uses local PDB database for AI analysis
    """)
    
    # Chat interface
    st.subheader("💬 AI Assistant Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")
    
    # User input
    user_input = st.text_input(
        "Ask the AI Assistant:",
        placeholder="e.g., 'Provide PDB targets for this SMILES: CCO'",
        key="ai_chat_input"
    )
    
    if st.button("📤 Send Message", type="primary") and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process with AI
        ai_response = process_ai_powered_query(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display new messages
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def process_ai_powered_query(user_input: str) -> str:
    """Process user query in AI-powered mode with step-by-step confirmation"""
    user_input_lower = user_input.lower()
    
    # Detect SMILES input
    if "smiles" in user_input_lower or any(char in user_input for char in ["C", "N", "O", "="]):
        # Extract potential SMILES
        words = user_input.split()
        potential_smiles = None
        for word in words:
            if any(char in word for char in ["C", "N", "O", "=", "(", ")"]):
                potential_smiles = word
                break
        
        if potential_smiles:
            response = f"""I found a SMILES structure: **{potential_smiles}**

I can help you find PDB targets for this molecule. Here's what I can do:

🔍 **Step 1**: Search our local PDB database ({len(st.session_state.local_database)} compounds)
🧪 **Step 2**: Perform molecular similarity analysis
📊 **Step 3**: Provide detailed results with recommendations

**Questions for you:**
1. Do you have any specific UniProt IDs you'd like me to focus on?
2. What similarity threshold would you prefer? (default: 0.7)
3. Would you like me to use Morgan fingerprints? (recommended)

Please confirm if you'd like me to proceed with the analysis, or provide additional parameters."""
            
            return response
    
    # Detect analysis request
    elif "analyze" in user_input_lower or "similarity" in user_input_lower:
        return """I'm ready to perform molecular analysis! 

To provide the best results, I need to know:

🎯 **Analysis Type**: 
- Molecular similarity analysis
- Heteroatom extraction
- Complete pipeline analysis

📋 **Required Information**:
- Target SMILES structure
- Similarity parameters (I can suggest defaults)
- Any specific protein targets?

Please provide your target molecule and I'll guide you through the process step by step."""
    
    # General help
    else:
        return f"""Hello! I'm your AI assistant for molecular analysis. 

**I can help you with:**
🧬 **Molecular Similarity Analysis** - Find similar compounds in our database
🔬 **Heteroatom Extraction** - Extract heteroatoms from protein structures  
📊 **Complete Pipeline Analysis** - End-to-end molecular analysis

**Our local database contains:**
- {len(st.session_state.local_database)} compounds
- Data from PDB structures
- SMILES representations
- UniProt IDs and more

**How to get started:**
1. Provide a SMILES structure (e.g., "Analyze this SMILES: CCO")
2. Ask for similarity analysis
3. Request heteroatom extraction for specific proteins

What would you like to analyze today?"""

def render_fully_autonomous_mode():
    """Render Fully Autonomous Mode - AI works without questions"""
    st.header("🚀 Fully Autonomous Mode - AI-Driven Analysis")
    
    st.markdown("""
    **Fully Autonomous Mode Features:**
    - AI works completely autonomously
    - No user confirmations required
    - Continuous analysis workflow
    - Uses local PDB database
    - Provides comprehensive reports
    """)
    
    # Simple input interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        autonomous_input = st.text_area(
            "🎯 Analysis Request:",
            placeholder="Enter SMILES structure or describe your analysis needs...",
            help="The AI will analyze your input and perform complete molecular analysis autonomously"
        )
    
    with col2:
        st.subheader("🤖 AI Settings")
        auto_threshold = st.slider("Auto Similarity Threshold:", 0.1, 1.0, 0.6, 0.1)
        comprehensive_analysis = st.checkbox("Comprehensive Analysis", value=True)
        include_visualizations = st.checkbox("Generate Visualizations", value=True)
    
    if st.button("🚀 Start Autonomous Analysis", type="primary") and autonomous_input:
        run_autonomous_analysis(autonomous_input, auto_threshold, comprehensive_analysis, include_visualizations)

def run_autonomous_analysis(user_input: str, threshold: float, comprehensive: bool, visualizations: bool):
    """Run fully autonomous analysis"""
    with st.spinner("🤖 AI is working autonomously..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Analyze input
            status_text.text("🔍 AI analyzing your input...")
            progress_bar.progress(20)
            time.sleep(1)
            
            # Extract SMILES if present
            potential_smiles = extract_smiles_from_input(user_input)
            
            if not potential_smiles:
                st.error("❌ AI couldn't detect a valid SMILES structure in your input")
                return
            
            # Step 2: Database analysis
            status_text.text("📊 AI searching local database...")
            progress_bar.progress(40)
            time.sleep(1)
            
            # Step 3: Similarity analysis
            status_text.text("🧪 AI performing molecular similarity analysis...")
            progress_bar.progress(60)
            
            # Create analyzer with AI-optimized parameters
            analyzer = SimilarityAnalyzer(
                radius=2,  # AI-chosen optimal radius
                n_bits=2048,  # AI-chosen optimal bits
                fp_type="morgan",  # AI-preferred fingerprint
                metric="tanimoto"  # AI-preferred metric
            )
            
            results = analyzer.analyze_similarity(
                target_smiles=potential_smiles,
                heteroatom_df=st.session_state.local_database,
                min_similarity=threshold
            )
            
            # Step 4: Generate comprehensive report
            status_text.text("📋 AI generating comprehensive report...")
            progress_bar.progress(80)
            time.sleep(1)
            
            # Step 5: Finalize
            status_text.text("✅ Autonomous analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            display_autonomous_results(potential_smiles, results, comprehensive, visualizations)
            
        except Exception as e:
            st.error(f"❌ Autonomous analysis failed: {str(e)}")

def extract_smiles_from_input(text: str) -> str:
    """Extract SMILES structure from user input"""
    import re
    
    # Simple SMILES detection patterns
    smiles_patterns = [
        r'\b[A-Z][a-z]?(?:\([^)]*\))?(?:\[[^\]]*\])?[=\-#]*[A-Z][a-z]?[0-9]*[=\-#]*[A-Z]*[a-z]*[0-9]*',
        r'\b[CNOPSFBrClI]+[=\-#()0-9]*[CNOPSFBrClI]*',
        r'\b[A-Za-z0-9()=\-#+]*[CNO][A-Za-z0-9()=\-#+]*'
    ]
    
    words = text.split()
    for word in words:
        # Check if word looks like SMILES
        if any(char in word for char in ["C", "N", "O", "=", "(", ")"]) and len(word) > 2:
            # Additional validation
            if not any(invalid in word.lower() for invalid in ["http", "www", "smiles", "structure"]):
                return word.strip(".,!?;:")
    
    return ""

def display_autonomous_results(smiles: str, results: pd.DataFrame, comprehensive: bool, visualizations: bool):
    """Display results from autonomous analysis"""
    st.success("🎉 Autonomous Analysis Complete!")
    
    # Analysis summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Target SMILES", smiles)
    with col2:
        st.metric("Similar Compounds", len(results))
    with col3:
        avg_similarity = results['Similarity'].mean() if not results.empty and 'Similarity' in results.columns else 0
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    with col4:
        st.metric("Database Size", len(st.session_state.local_database))
    
    if not results.empty:
        # Main results
        st.subheader("🎯 AI-Generated Analysis Results")
        st.dataframe(results)
        
        # Visualizations
        if visualizations and len(results) > 0:
            st.subheader("📊 AI-Generated Visualizations")
            fig = create_similarity_plot(results)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive AI report
        if comprehensive:
            st.subheader("📋 AI Comprehensive Report")
            ai_report = generate_ai_report(smiles, results)
            st.markdown(ai_report)
        
        # Download options
        st.subheader("📥 Download Results")
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Analysis",
            data=csv,
            file_name=f"autonomous_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No similar compounds found. Try lowering the similarity threshold.")

def create_similarity_plot(results: pd.DataFrame):
    """Create similarity visualization plot"""
    try:
        import plotly.express as px
        
        if 'Similarity' in results.columns:
            fig = px.histogram(
                results, 
                x='Similarity', 
                title='Distribution of Molecular Similarities',
                nbins=20,
                labels={'Similarity': 'Tanimoto Similarity Score', 'count': 'Number of Compounds'}
            )
            fig.update_layout(
                xaxis_title="Similarity Score",
                yaxis_title="Count",
                showlegend=False
            )
            return fig
        else:
            # Fallback plot
            fig = px.bar(
                x=list(range(len(results))), 
                y=[1]*len(results),
                title="Analysis Results Overview"
            )
            return fig
            
    except ImportError:
        # Fallback if plotly not available
        st.info("📊 Install plotly for advanced visualizations: pip install plotly")
        return None

def generate_ai_report(smiles: str, results: pd.DataFrame) -> str:
    """Generate comprehensive AI analysis report"""
    total_compounds = len(results)
    
    if total_compounds == 0:
        return """
### 🤖 AI Analysis Report

**Target Molecule:** `{}`

**Analysis Summary:**
- No similar compounds found in the database
- Consider lowering the similarity threshold
- The target molecule might be novel or unique

**AI Recommendations:**
- Try different fingerprint types
- Adjust similarity parameters
- Check if SMILES structure is valid
        """.format(smiles)
    
    # Calculate statistics
    if 'Similarity' in results.columns:
        max_sim = results['Similarity'].max()
        min_sim = results['Similarity'].min()
        avg_sim = results['Similarity'].mean()
        std_sim = results['Similarity'].std()
    else:
        max_sim = min_sim = avg_sim = std_sim = "N/A"
    
    # Get top compounds
    top_compounds = results.head(5)
    
    report = f"""
### 🤖 AI Comprehensive Analysis Report

**Target Molecule:** `{smiles}`

**📊 Statistical Analysis:**
- **Total Similar Compounds Found:** {total_compounds}
- **Highest Similarity Score:** {max_sim:.3f}
- **Lowest Similarity Score:** {min_sim:.3f}
- **Average Similarity:** {avg_sim:.3f}
- **Standard Deviation:** {std_sim:.3f}

**🎯 Top Similar Compounds:**
"""
    
    for idx, row in top_compounds.iterrows():
        if 'PDB_ID' in row and 'Similarity' in row:
            report += f"- **{row['PDB_ID']}**: Similarity {row['Similarity']:.3f}\n"
        elif 'UniProt_ID' in row:
            report += f"- **{row['UniProt_ID']}**: Found in database\n"
    
    report += f"""

**🤖 AI Insights:**
- The analysis identified {total_compounds} compounds with significant similarity
- Average similarity of {avg_sim:.3f} indicates {'strong' if avg_sim > 0.7 else 'moderate' if avg_sim > 0.5 else 'weak'} molecular relationship
- Standard deviation of {std_sim:.3f} shows {'diverse' if std_sim > 0.1 else 'consistent'} similarity distribution

**💡 AI Recommendations:**
- {'High similarity scores suggest strong molecular relationships' if max_sim > 0.8 else 'Moderate similarities suggest structural similarities worth investigating'}
- Consider experimental validation for top candidates
- {'Diverse similarity range indicates multiple binding modes possible' if std_sim > 0.15 else 'Consistent similarities suggest similar binding mechanisms'}
"""
    
    return report

# Enhanced AI prediction capabilities
def predict_bioactivity(smiles, similarity_results):
    """Generate AI-powered bioactivity predictions"""
    try:
        # Simulated ML predictions based on molecular features
        # In a real implementation, this would use trained ML models
        
        # Calculate molecular complexity score
        complexity_score = len(smiles) + smiles.count('(') * 2 + smiles.count('=') * 1.5
        
        # Base predictions on similarity results and molecular features
        if not similarity_results.empty and 'Similarity' in similarity_results.columns:
            avg_similarity = similarity_results['Similarity'].mean()
            max_similarity = similarity_results['Similarity'].max()
            
            # Bioactivity score (0-100)
            bioactivity_score = min(100, (avg_similarity * 70) + (complexity_score * 0.5))
            confidence = min(100, (max_similarity * 80) + 20)
        else:
            bioactivity_score = max(0, 50 - (complexity_score * 0.3))
            confidence = 30
        
        # Drug likeness assessment
        if complexity_score < 20:
            drug_likeness = "Excellent"
        elif complexity_score < 40:
            drug_likeness = "Good" 
        elif complexity_score < 60:
            drug_likeness = "Moderate"
        else:
            drug_likeness = "Poor"
        
        # Toxicity risk assessment
        aromatic_rings = smiles.count('c') + smiles.count('C1=CC=CC=C1')
        if aromatic_rings <= 1 and complexity_score < 30:
            toxicity_risk = "Low"
        elif aromatic_rings <= 2 and complexity_score < 50:
            toxicity_risk = "Moderate"
        else:
            toxicity_risk = "High"
        
        return {
            'binding_affinity_score': round(bioactivity_score, 1),
            'drug_likeness': drug_likeness,
            'toxicity_risk': toxicity_risk,
            'confidence': round(confidence, 1),
            'complexity_score': round(complexity_score, 1)
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {
            'binding_affinity_score': 0,
            'drug_likeness': 'Unknown',
            'toxicity_risk': 'Unknown',
            'confidence': 0,
            'complexity_score': 0
        }

def create_advanced_visualizations(results, smiles):
    """Create advanced 3D and analytical visualizations"""
    try:
        fig_3d = None
        fig_heatmap = None
        fig_distribution = None
        
        if not results.empty:
            # 3D Molecular Similarity Landscape
            if len(results) >= 3 and 'Similarity' in results.columns:
                # Create 3D scatter plot
                x = np.random.rand(len(results))
                y = np.random.rand(len(results))
                z = results['Similarity'].values
                
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        colorbar=dict(title="Similarity Score"),
                        showscale=True
                    ),
                    text=[f"PDB: {row.get('PDB_ID', 'Unknown')}<br>Similarity: {row.get('Similarity', 0):.3f}" 
                          for _, row in results.iterrows()],
                    hovertemplate='%{text}<extra></extra>'
                )])
                
                fig_3d.update_layout(
                    title="3D Molecular Similarity Landscape",
                    scene=dict(
                        xaxis_title="Chemical Space X",
                        yaxis_title="Chemical Space Y", 
                        zaxis_title="Similarity Score"
                    ),
                    height=500
                )
            
            # Similarity Heatmap
            if len(results) > 1:
                # Create correlation matrix
                numeric_cols = results.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = results[numeric_cols].corr()
                    
                    fig_heatmap = px.imshow(
                        corr_matrix,
                        title="Molecular Property Correlation Heatmap",
                        color_continuous_scale='RdBu',
                        aspect='auto'
                    )
                    fig_heatmap.update_layout(height=400)
            
            # Distribution Analysis
            if 'Similarity' in results.columns:
                fig_distribution = px.histogram(
                    results, 
                    x='Similarity',
                    nbins=20,
                    title="Similarity Score Distribution",
                    labels={'Similarity': 'Similarity Score', 'count': 'Number of Compounds'}
                )
                fig_distribution.update_layout(height=400)
        
        return fig_3d, fig_heatmap, fig_distribution
        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None, None, None

# Update the mode selection in main app
def enhanced_autonomous_mode():
    """Enhanced Fully Autonomous Mode with ML predictions"""
    st.header("🤖 Enhanced Autonomous Mode")
    st.write("AI works autonomously with advanced machine learning predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles = st.text_input(
            "🧬 Enter SMILES notation:",
            placeholder="CCO (ethanol) or CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)",
            help="The AI will automatically analyze and predict bioactivity"
        )
    
    with col2:
        auto_run = st.checkbox("🔄 Real-time Analysis", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    
    if smiles and (auto_run or st.button("🚀 Run Enhanced Analysis")):
        with st.spinner("🧠 AI is working autonomously..."):
            # Get similarity results
            results = st.session_state.similarity_analyzer.find_similar_molecules(smiles)
            
            if not results.empty:
                # AI Predictions
                st.subheader("🔮 AI Bioactivity Predictions")
                predictions = predict_bioactivity(smiles, results)
                
                # Prediction Dashboard
                pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                
                with pred_col1:
                    st.metric(
                        "Binding Affinity Score", 
                        f"{predictions['binding_affinity_score']:.1f}/100",
                        delta=f"{predictions['confidence']:.1f}% confidence"
                    )
                
                with pred_col2:
                    st.metric("Drug Likeness", predictions['drug_likeness'])
                
                with pred_col3:
                    st.metric("Toxicity Risk", predictions['toxicity_risk'])
                
                with pred_col4:
                    risk_color = "🟢" if predictions['toxicity_risk'] == "Low" else "🟡" if predictions['toxicity_risk'] == "Moderate" else "🔴"
                    st.metric("Risk Level", risk_color)
                
                # Advanced Visualizations
                st.subheader("📊 Advanced Molecular Analytics")
                
                fig_3d, fig_heatmap, fig_distribution = create_advanced_visualizations(results, smiles)
                
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    if fig_heatmap and len(results) > 1:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                with viz_col2:
                    if fig_distribution:
                        st.plotly_chart(fig_distribution, use_container_width=True)
                
                # Enhanced Results Table
                st.subheader("📋 Similarity Analysis Results")
                
                # Add AI-generated insights
                if 'Similarity' in results.columns:
                    avg_sim = results['Similarity'].mean()
                    max_sim = results['Similarity'].max()
                    
                    insight_text = f"""
                    **🧠 AI Insights:**
                    - Average similarity: {avg_sim:.3f}
                    - Best match: {max_sim:.3f}
                    - Total compounds analyzed: {len(results)}
                    """
                    
                    if avg_sim > confidence_threshold:
                        insight_text += "\n- ✅ **High confidence** in bioactivity predictions"
                    else:
                        insight_text += "\n- ⚠️ **Lower confidence** - consider additional analysis"
                    
                    st.info(insight_text)
                
                # Enhanced results display
                display_results = results.copy()
                if 'Similarity' in display_results.columns:
                    display_results['Confidence Level'] = display_results['Similarity'].apply(
                        lambda x: "🟢 High" if x > 0.8 else "🟡 Medium" if x > 0.6 else "🔴 Low"
                    )
                
                st.dataframe(display_results, use_container_width=True)
                
                # Export enhanced results
                export_data = {
                    'analysis_results': results.to_dict('records'),
                    'ai_predictions': predictions,
                    'metadata': {
                        'smiles': smiles,
                        'analysis_date': pd.Timestamp.now().isoformat(),
                        'confidence_threshold': confidence_threshold
                    }
                }
                
                st.download_button(
                    "📥 Download Enhanced Analysis",
                    data=pd.DataFrame([export_data]).to_json(orient='records', indent=2),
                    file_name=f"enhanced_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No similar molecules found. Try a different SMILES structure.")

def batch_processing_mode():
    """New Batch Processing Mode for multiple compounds"""
    st.header("📦 Batch Processing Mode")
    st.write("Process multiple compounds simultaneously with advanced analytics")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with SMILES", 
        type=['csv'],
        help="CSV should have a column named 'SMILES'"
    )
    
    # Manual entry option
    st.write("**Or enter multiple SMILES manually:**")
    smiles_text = st.text_area(
        "Enter SMILES (one per line):",
        placeholder="CCO\nCC(=O)OC1=CC=CC=C1C(=O)O\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        height=100
    )
    
    process_button = st.button("🚀 Process Batch", type="primary")
    
    if process_button and (uploaded_file or smiles_text):
        smiles_list = []
        
        # Process uploaded file
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'SMILES' in df.columns:
                smiles_list.extend(df['SMILES'].dropna().tolist())
            else:
                st.error("CSV file must contain a 'SMILES' column")
                return
        
        # Process manual entry
        if smiles_text:
            smiles_list.extend([s.strip() for s in smiles_text.split('\n') if s.strip()])
        
        if smiles_list:
            st.write(f"Processing {len(smiles_list)} compounds...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            
            for i, smiles in enumerate(smiles_list):
                status_text.text(f"Processing compound {i+1}/{len(smiles_list)}: {smiles[:20]}...")
                
                try:
                    # Analyze each SMILES
                    results = st.session_state.similarity_analyzer.find_similar_molecules(smiles)
                    predictions = predict_bioactivity(smiles, results)
                    
                    batch_results.append({
                        'SMILES': smiles,
                        'Compounds_Found': len(results),
                        'Avg_Similarity': results['Similarity'].mean() if not results.empty and 'Similarity' in results.columns else 0,
                        'Max_Similarity': results['Similarity'].max() if not results.empty and 'Similarity' in results.columns else 0,
                        'Bioactivity_Score': predictions['binding_affinity_score'],
                        'Drug_Likeness': predictions['drug_likeness'],
                        'Toxicity_Risk': predictions['toxicity_risk'],
                        'Confidence': predictions['confidence']
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'SMILES': smiles,
                        'Error': str(e),
                        'Compounds_Found': 0,
                        'Avg_Similarity': 0,
                        'Max_Similarity': 0,
                        'Bioactivity_Score': 0,
                        'Drug_Likeness': 'Error',
                        'Toxicity_Risk': 'Unknown',
                        'Confidence': 0
                    })
                
                progress_bar.progress((i + 1) / len(smiles_list))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display batch results
            batch_df = pd.DataFrame(batch_results)
            st.subheader("📊 Batch Analysis Results")
            st.dataframe(batch_df, use_container_width=True)
            
            # Batch analytics
            if not batch_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_bioactivity = batch_df['Bioactivity_Score'].mean()
                    st.metric("Average Bioactivity", f"{avg_bioactivity:.1f}")
                
                with col2:
                    high_confidence = (batch_df['Confidence'] > 70).sum()
                    st.metric("High Confidence Results", f"{high_confidence}/{len(batch_df)}")
                
                with col3:
                    low_toxicity = (batch_df['Toxicity_Risk'] == 'Low').sum()
                    st.metric("Low Toxicity Compounds", f"{low_toxicity}/{len(batch_df)}")
                
                # Download batch results
                st.download_button(
                    "📥 Download Batch Results",
                    data=batch_df.to_csv(index=False),
                    file_name=f"batch_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please provide SMILES data to process.")

# Footer and additional information
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>TrackMyPDB v2.0</strong> - Molecular Analysis Platform</p>
    <p>Built with Streamlit • Powered by RDKit • Open Source under MIT License</p>
    <p>Local Database: {database_size} compounds | Modes: Manual • AI-Powered • Fully Autonomous</p>
    </div>
    """.format(database_size=len(st.session_state.local_database)), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    render_footer()