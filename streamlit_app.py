"""
TrackMyPDB - Streamlit Application
@author Anu Gamage

A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures
and finding molecularly similar compounds using advanced fingerprint-based similarity analysis.

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

# Import the new agentic layer components
from backend.agentic_layer import (
    TrackMyPDBAgenticInterface, 
    AgentMode, 
    AnalysisType,
    RDKIT_AVAILABLE
)

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
try:
    from backend.heteroatom_extractor import HeteroatomExtractor
    # Import RDKit first to ensure it's available
    if RDKIT_AVAILABLE:
        import rdkit
        from backend.similarity_analyzer import SimilarityAnalyzer
    else:
        from backend.similarity_analyzer_simple import MolecularSimilarityAnalyzer as SimilarityAnalyzer
        st.warning("⚠️ RDKit not available - using simplified molecular similarity")
except ImportError as e:
    # Only fall back to simplified version if RDKit specifically fails
    if 'rdkit' in str(e):
        from backend.similarity_analyzer_simple import MolecularSimilarityAnalyzer as SimilarityAnalyzer
        st.warning("⚠️ RDKit not available - using simplified molecular similarity")
    else:
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

@st.cache_resource
def initialize_agentic_interface():
    return TrackMyPDBAgenticInterface()

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()

if "agentic_interface" not in st.session_state:
    st.session_state.agentic_interface = initialize_agentic_interface()

if "nl_interface" not in st.session_state:
    st.session_state.nl_interface = NaturalLanguageInterface(st.session_state.agent)

def main():
    st.title("🧬 TrackMyPDB - AI-Powered Analysis")
    
    # Display system status
    display_system_status()
    
    # Sidebar with analysis mode selection
    st.sidebar.title("Analysis Mode")
    mode = st.sidebar.selectbox(
        "Choose your interaction mode:",
        ["🤖 AI-Powered Agentic Analysis", "💬 Natural Language", "📊 Traditional Interface"]
    )
    
    if mode == "🤖 AI-Powered Agentic Analysis":
        render_agentic_interface()
    elif mode == "💬 Natural Language":
        st.session_state.nl_interface.render_chat_interface()
    else:
        render_traditional_interface()

def display_system_status():
    """Display system capabilities and status"""
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "RDKit Status", 
                "✅ Available" if RDKIT_AVAILABLE else "❌ Not Available",
                help="RDKit provides advanced molecular analysis capabilities"
            )
        
        with col2:
            agent_status = st.session_state.agentic_interface.get_agent_status()
            ai_agents = sum(1 for status in agent_status.values() if status['ai_enabled'])
            st.metric(
                "AI Agents", 
                f"{ai_agents}/{len(agent_status)} Active",
                help="Number of AI-powered analysis agents available"
            )
        
        with col3:
            available_analyses = len(st.session_state.agentic_interface.get_available_analysis_types())
            st.metric(
                "Analysis Types", 
                available_analyses,
                help="Number of different molecular analysis methods available"
            )

def render_agentic_interface():
    """Render the new AI-powered agentic analysis interface"""
    st.header("🤖 AI-Powered Agentic Molecular Analysis")
    
    st.markdown("""
    This advanced interface uses AI agents to perform comprehensive molecular analysis with:
    - **Multi-agent coordination** for parallel analysis
    - **AI-powered insights** using Google Gemini
    - **Adaptive analysis** based on molecular characteristics
    - **Comprehensive reporting** with recommendations
    """)
    
    # Analysis configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_smiles = st.text_area(
            "🧪 Target SMILES String",
            placeholder="Enter SMILES string (e.g., CCO for ethanol)",
            help="Enter the SMILES representation of your target molecule"
        )
    
    with col2:
        # Agent mode selection
        agent_mode = st.selectbox(
            "🤖 AI Mode",
            options=["Manual", "AI Assisted", "Fully Autonomous"],
            index=1,
            help=""":
            - **Manual**: Traditional analysis without AI insights
            - **AI Assisted**: AI provides insights and recommendations
            - **Fully Autonomous**: AI makes analysis decisions
            """
        )
        
        # Convert to enum
        mode_mapping = {
            "Manual": AgentMode.MANUAL,
            "AI Assisted": AgentMode.AI_ASSISTED,
            "Fully Autonomous": AgentMode.FULLY_AUTONOMOUS
        }
        selected_mode = mode_mapping[agent_mode]
    
    # Analysis type selection
    st.subheader("🔬 Analysis Configuration")
    
    available_types = st.session_state.agentic_interface.get_available_analysis_types()
    analysis_options = {
        "Morgan Fingerprint Similarity": AnalysisType.MORGAN_SIMILARITY,
        "Tanimoto Similarity Analysis": AnalysisType.TANIMOTO_SIMILARITY,
        "Drug-Likeness Assessment": AnalysisType.DRUG_LIKENESS
    }
    
    selected_analyses = st.multiselect(
        "Select Analysis Types",
        options=list(analysis_options.keys()),
        default=["Morgan Fingerprint Similarity", "Drug-Likeness Assessment"],
        help="Choose which types of molecular analysis to perform"
    )
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius = st.slider("Morgan Radius", 1, 4, 2)
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)
        
        with col2:
            n_bits = st.selectbox("Fingerprint Bits", [1024, 2048, 4096], index=1)
            fingerprint_type = st.selectbox("Fingerprint Type", ["morgan", "maccs", "atompair"])
        
        with col3:
            context = st.text_area(
                "Analysis Context",
                placeholder="Optional: Provide context for AI analysis",
                height=100
            )
    
    # File upload for heteroatom data
    uploaded_file = st.file_uploader(
        "📁 Upload Heteroatom Data (Optional)",
        type=['csv'],
        help="Upload existing heteroatom extraction results to use in similarity analysis"
    )
    
    heteroatom_data = None
    if uploaded_file is not None:
        try:
            heteroatom_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(heteroatom_data)} compounds from uploaded file")
            with st.expander("📊 Data Preview"):
                st.dataframe(heteroatom_data.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif hasattr(st.session_state, 'heteroatom_results') and st.session_state.heteroatom_results is not None:
        heteroatom_data = st.session_state.heteroatom_results
        st.info("ℹ️ Using heteroatom data from previous extraction")
    
    # Run analysis button
    if st.button("🚀 Run AI-Powered Analysis", type="primary"):
        if not target_smiles:
            st.error("❌ Please enter a SMILES string")
            return
        
        if not selected_analyses:
            st.error("❌ Please select at least one analysis type")
            return
        
        # Convert selected analyses to enums
        analysis_types = [analysis_options[analysis] for analysis in selected_analyses]
        
        # Run the agentic analysis
        with st.spinner("🤖 AI agents are analyzing your molecule..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Simulate progress updates
                for i in range(0, 101, 20):
                    progress_bar.progress(i)
                    if i == 0:
                        status_text.text("🔍 Initializing AI agents...")
                    elif i == 20:
                        status_text.text("🧪 Analyzing molecular structure...")
                    elif i == 40:
                        status_text.text("🔬 Computing similarity fingerprints...")
                    elif i == 60:
                        status_text.text("📊 Evaluating drug-likeness properties...")
                    elif i == 80:
                        status_text.text("🤖 Generating AI insights...")
                    elif i == 100:
                        status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                
                # Execute the agentic analysis
                results = st.session_state.agent.execute_action(
                    "agentic_analysis",
                    {
                        "target_smiles": target_smiles,
                        "mode": selected_mode.value,
                        "analysis_types": [at.value for at in analysis_types],
                        "radius": radius,
                        "n_bits": n_bits,
                        "threshold": threshold,
                        "fingerprint_type": fingerprint_type,
                        "context": context,
                        "heteroatom_data": heteroatom_data
                    }
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Display results
                if "error" in results:
                    st.error(f"❌ Analysis failed: {results['error']}")
                else:
                    display_agentic_results(results["agentic_results"], target_smiles)
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

def display_agentic_results(results, target_smiles):
    """Display comprehensive agentic analysis results"""
    st.success("🎉 AI-Powered Analysis Complete!")
    
    # Display analysis summary
    summary = results["analysis_summary"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analyses Run", f"{summary['analyses_successful']}/{summary['analyses_requested']}")
    with col2:
        st.metric("Avg Confidence", f"{summary['average_confidence']:.2f}")
    with col3:
        st.metric("Execution Time", f"{summary['total_execution_time']:.1f}s")
    with col4:
        st.metric("Mode", summary['mode'].replace('_', ' ').title())
    
    # Display comprehensive AI report
    if "comprehensive_report" in results:
        st.subheader("📋 AI-Generated Comprehensive Report")
        with st.expander("🤖 View Full AI Analysis Report", expanded=True):
            st.markdown(results["comprehensive_report"])
    
    # Display individual analysis results
    st.subheader("🔬 Detailed Analysis Results")
    
    for analysis_type, result in results["results"].items():
        with st.expander(f"📊 {analysis_type.replace('_', ' ').title()} Analysis", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Status indicators
                status = "✅ Success" if result.success else "❌ Failed"
                st.metric("Status", status)
                st.metric("Confidence", f"{result.confidence:.2f}")
                st.metric("Time", f"{result.execution_time:.2f}s")
            
            with col1:
                if result.success:
                    # Display analysis-specific results
                    if analysis_type == "morgan_similarity":
                        display_similarity_results(result.results, "Morgan")
                    elif analysis_type == "tanimoto_similarity":
                        display_similarity_results(result.results, "Tanimoto")
                    elif analysis_type == "drug_likeness":
                        display_drug_likeness_results(result.results)
                    
                    # Display recommendations
                    if result.recommendations:
                        st.markdown("**🎯 Key Recommendations:**")
                        for rec in result.recommendations:
                            st.markdown(f"• {rec}")
                    
                    # Display AI insights if available
                    if result.results.get('ai_insights'):
                        with st.expander("🤖 AI Insights", expanded=False):
                            st.markdown(result.results['ai_insights'])
                else:
                    st.error(f"Analysis failed: {result.error_message}")

def display_similarity_results(results, analysis_type):
    """Display similarity analysis results"""
    similarities = results.get('similarities', [])
    
    if similarities:
        st.markdown(f"**Found {len(similarities)} similar compounds**")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(similarities[:10])  # Show top 10
        
        # Configure columns for display
        if 'similarity' in df.columns:
            similarity_col = 'similarity'
        elif 'tanimoto_similarity' in df.columns:
            similarity_col = 'tanimoto_similarity'
        else:
            similarity_col = None
        
        if similarity_col:
            df = df.sort_values(similarity_col, ascending=False)
            
            # Display interactive table
            st.dataframe(
                df,
                column_config={
                    similarity_col: st.column_config.ProgressColumn(
                        f"{analysis_type} Similarity",
                        help="Molecular similarity score",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    'molecular_weight': st.column_config.NumberColumn(
                        "MW",
                        help="Molecular Weight",
                        format="%.1f"
                    ),
                    'logp': st.column_config.NumberColumn(
                        "LogP",
                        help="Lipophilicity",
                        format="%.2f"
                    )
                },
                use_container_width=True
            )
    else:
        st.warning("No similar compounds found above the threshold")

def display_drug_likeness_results(results):
    """Display drug-likeness analysis results"""
    properties = results.get('properties', {})
    lipinski = results.get('lipinski_assessment', {})
    score = results.get('drug_likeness_score', 0)
    
    # Display drug-likeness score
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Drug-Likeness Score", 
            f"{score:.2f}",
            help="Overall drug-likeness assessment (0-1)"
        )
    
    with col2:
        lipinski_status = "✅ Pass" if lipinski.get('passed', False) else "❌ Fail"
        st.metric(
            "Lipinski's Rule", 
            lipinski_status,
            f"{lipinski.get('violations', 0)} violations"
        )
    
    # Display molecular properties
    if properties:
        st.markdown("**Molecular Properties:**")
        prop_cols = st.columns(4)
        
        prop_items = list(properties.items())
        for i, (prop, value) in enumerate(prop_items[:4]):
            with prop_cols[i % 4]:
                if isinstance(value, (int, float)):
                    st.metric(prop.replace('_', ' ').title(), f"{value:.2f}")

def render_traditional_interface():
    """Render the traditional interface with advanced analysis options"""
    st.subheader("📊 Traditional Analysis Interface")
    
    if 'heteroatom_results' not in st.session_state:
        st.session_state.heteroatom_results = None
    
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
                    st.session_state.heteroatom_results = results["results"]
                    st.write(results["results"])
    
    elif analysis_type == "🧪 Similarity Analysis":
        st.subheader("Similarity Analysis")
        
        if st.session_state.heteroatom_results is None:
            st.warning("⚠️ Please run heteroatom extraction first")
            return
        
        # Advanced molecular analysis options
        with st.expander("🔬 Advanced Analysis Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                if hasattr(SimilarityAnalyzer, 'FINGERPRINT_TYPES'):
                    fp_type = st.selectbox(
                        "Fingerprint Type",
                        options=list(SimilarityAnalyzer.FINGERPRINT_TYPES.keys()),
                        help=""":
                        - Morgan: Extended connectivity fingerprints (ECFP)
                        - MACCS: 166 predefined structural keys
                        - Topological: Atom pair fingerprints
                        - RDKit: RDKit's default fingerprints
                        - Pattern: Detailed atom pair patterns
                        """
                    )
                    
                    similarity_metric = st.selectbox(
                        "Similarity Metric",
                        options=list(SimilarityAnalyzer.SIMILARITY_METRICS.keys()),
                        help=""":
                        - Tanimoto: Standard similarity coefficient
                        - Dice: Emphasizes common features
                        - Cosine: Angular similarity between fingerprints
                        """
                    )
                else:
                    fp_type = "morgan"
                    similarity_metric = "tanimoto"
                    st.info("Using simplified molecular analysis (RDKit not available)")
            
            with col2:
                if fp_type == 'morgan':
                    radius = st.slider(
                        "Morgan Fingerprint Radius",
                        min_value=1,
                        max_value=4,
                        value=2,
                        help="Radius parameter for Morgan fingerprint generation"
                    )
                    n_bits = st.slider(
                        "Number of Bits",
                        min_value=512,
                        max_value=4096,
                        value=2048,
                        step=512,
                        help="Number of bits in fingerprint"
                    )
                else:
                    radius = 2
                    n_bits = 2048
        
        smiles_input = st.text_area(
            "SMILES String",
            help="💡 AI Tip: Enter a SMILES string to find similar compounds"
        )
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Higher values mean more similar compounds"
        )
        
        if st.button("🔍 Analyze Similarity"):
            if smiles_input:
                with st.spinner("🧪 Analyzing similarity..."):
                    # Create analyzer with selected options
                    analyzer = SimilarityAnalyzer(
                        radius=radius,
                        n_bits=n_bits,
                        fp_type=fp_type,
                        metric=similarity_metric
                    )
                    results = analyzer.analyze_similarity(
                        target_smiles=smiles_input,
                        heteroatom_df=st.session_state.heteroatom_results,
                        min_similarity=threshold
                    )
                    st.success("✅ Analysis complete!")
                    
                    # Show advanced molecular information
                    if len(results) > 0:
                        with st.expander("🔬 Advanced Molecular Analysis", expanded=True):
                            st.markdown("### Molecular Property Differences")
                            property_cols = [col for col in results.columns if col.startswith('Delta_')]
                            if property_cols:
                                property_df = results[['PDB_ID', 'Heteroatom_Code'] + property_cols].head(10)
                                st.dataframe(
                                    property_df,
                                    use_container_width=True,
                                    column_config={
                                        prop: st.column_config.NumberColumn(
                                            prop.replace('Delta_', 'Δ '),
                                            help="Difference from target molecule",
                                            format="%.2f"
                                        ) for prop in property_cols
                                    }
                                )
                            
                            st.markdown("### Substructure Analysis")
                            substructure_df = results[['PDB_ID', 'Heteroatom_Code', 'Has_Substructure_Match', 'Substructure_Match_Count']].head(10)
                            st.dataframe(substructure_df, use_container_width=True)
    
    else:  # Complete Pipeline
        st.subheader("Complete Pipeline Analysis")
        
        # Advanced options in pipeline mode
        with st.expander("🔬 Advanced Analysis Options", expanded=True):
            if hasattr(SimilarityAnalyzer, 'FINGERPRINT_TYPES'):
                fp_type = st.selectbox(
                    "Fingerprint Type",
                    options=list(SimilarityAnalyzer.FINGERPRINT_TYPES.keys())
                )
                similarity_metric = st.selectbox(
                    "Similarity Metric",
                    options=list(SimilarityAnalyzer.SIMILARITY_METRICS.keys())
                )
            else:
                fp_type = "morgan"
                similarity_metric = "tanimoto"
        
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
            help="Higher values mean more similar compounds"
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
                            "threshold": threshold,
                            "fp_type": fp_type,
                            "metric": similarity_metric
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