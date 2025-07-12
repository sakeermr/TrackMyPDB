"""
TrackMyPDB Natural Language Interface
@author Anu Gamage

Three distinct operational modes:
1. Manual Mode - Real-time PDB fetching with manual inputs
2. AI-Powered Mode - Guided chatbot with step-by-step questions  
3. Fully Autonomous Mode - AI-driven analysis without user prompts

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
        
        # Initialize chat history for AI modes
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_analysis_state" not in st.session_state:
            st.session_state.current_analysis_state = {}
    
    def _load_local_database(self) -> pd.DataFrame:
        """
        Load PDB-derived Excel data from the Data module
        
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
                    st.info(f"Loaded {len(df)} records from {file}")
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                st.success(f"📊 Local database loaded: {len(combined_df)} total records from {len(all_data)} files")
                return combined_df
            else:
                st.warning("⚠️ No data files found in data module")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Error loading local database: {e}")
            return pd.DataFrame()
    
    def render_manual_mode_interface(self):
        """
        Mode 1: Manual Mode - Real-time PDB fetching with manual parameter input
        """
        st.header("🔧 Manual Mode - Real-time PDB Analysis")
        
        st.markdown("""
        **Manual Mode Features:**
        - 🔍 **Real-time PDB Data Bank fetching** for each UniProt ID
        - 📝 **Manual parameter configuration** for Morgan fingerprints
        - ⚙️ **User-controlled analysis settings** (radius, bit-vector length, thresholds)
        - 🎯 **Three processing stages**: Heteroatom Extraction → Similarity Analysis → Combined Pipeline
        - 📊 **Immediate results** with no internal step-by-step display
        """)
        
        # Input Configuration Section
        with st.expander("📋 Input Configuration", expanded=True):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🧬 Molecular Inputs")
                
                # UniProt IDs input
                uniprot_input = st.text_area(
                    "UniProt IDs (comma-separated):",
                    placeholder="P21554, P18031, P34972",
                    help="Enter UniProt IDs for real-time PDB structure fetching",
                    height=100
                )
                
                # SMILES input
                target_smiles = st.text_input(
                    "Target SMILES Structure:",
                    placeholder="CCO, c1ccccc1, CC(=O)O",
                    help="Enter the SMILES string of your target molecule for similarity analysis"
                )
            
            with col2:
                st.subheader("⚙️ Morgan Fingerprint Parameters")
                
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
        st.subheader("🎯 Analysis Stage Selection")
        analysis_stage = st.radio(
            "Choose processing stage:",
            [
                "Heteroatom Extraction Only",
                "Molecular Similarity Analysis Only", 
                "Combined Pipeline (Heteroatom + Similarity)"
            ],
            help="Select which analysis stage to execute"
        )
        
        # Execution Button
        if st.button("🚀 Execute Manual Analysis (Real-time PDB)", type="primary", use_container_width=True):
            if analysis_stage == "Heteroatom Extraction Only":
                if not uniprot_input.strip():
                    st.error("❌ UniProt IDs required for heteroatom extraction")
                    return
                self._execute_manual_heteroatom_extraction(uniprot_input)
                
            elif analysis_stage == "Molecular Similarity Analysis Only":
                if not target_smiles.strip():
                    st.error("❌ Target SMILES required for similarity analysis")
                    return
                self._execute_manual_similarity_analysis(target_smiles, radius, n_bits, threshold, fp_type, metric)
                
            else:  # Combined Pipeline
                if not uniprot_input.strip() or not target_smiles.strip():
                    st.error("❌ Both UniProt IDs and target SMILES required for combined pipeline")
                    return
                self._execute_manual_combined_pipeline(uniprot_input, target_smiles, radius, n_bits, threshold, fp_type, metric)
    
    def render_ai_powered_mode_interface(self):
        """
        Mode 2: AI-Powered Mode - Guided chatbot with clarifying questions
        """
        st.header("🤖 AI-Powered Mode - Interactive Guided Analysis")
        
        st.markdown("""
        **AI-Powered Mode Features:**
        - 💬 **Chatbot-style dialog** for guided analysis
        - ❓ **Clarifying questions** at each decision point
        - ✅ **User confirmation** before proceeding with parameters
        - 📊 **Excel-based analysis** using local PDB-derived database
        - 🔮 **Future enhancement**: Live PDB connectivity
        """)
        
        # Database Info
        with st.expander("📊 Current Data Source", expanded=False):
            if not self.local_database.empty:
                st.info(f"**Local Database**: {len(self.local_database)} records from Excel files")
                st.write(f"- **UniProt IDs**: {self.local_database['UniProt_ID'].nunique()}")
                st.write(f"- **PDB Structures**: {self.local_database['PDB_ID'].nunique()}")
                st.write(f"- **Heteroatoms**: {self.local_database['Heteroatom_Code'].nunique()}")
            else:
                st.warning("⚠️ Local database not available")
        
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
            ai_response = self._generate_ai_guided_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            st.rerun()
    
    def render_fully_autonomous_mode_interface(self):
        """
        Mode 3: Fully Autonomous Mode - AI processes without user prompts
        """
        st.header("🚀 Fully Autonomous Mode - AI-Driven Analysis")
        
        st.markdown("""
        **Fully Autonomous Mode Features:**
        - 🧠 **Complete AI automation** after initial request
        - 🔄 **Continuous processing** of new requests
        - ⚙️ **Automatic parameter determination** for all analyses
        - 📊 **Excel-based processing** using local PDB-derived database
        - 📈 **Comprehensive autonomous reports** with visualizations
        """)
        
        # Configuration
        with st.expander("🤖 Autonomous AI Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                auto_threshold = st.slider("Auto Similarity Threshold:", 0.1, 1.0, 0.6, 0.1)
                comprehensive_analysis = st.checkbox("Comprehensive Analysis", value=True)
            
            with col2:
                include_visualizations = st.checkbox("Generate Visualizations", value=True)
                auto_download = st.checkbox("Auto-generate Downloads", value=True)
            
            with col3:
                processing_speed = st.selectbox("Processing Speed:", ["Fast", "Balanced", "Thorough"], index=1)
                result_detail = st.selectbox("Result Detail:", ["Summary", "Standard", "Detailed"], index=1)
        
        # Input Interface
        st.subheader("🎯 Analysis Request")
        autonomous_input = st.text_area(
            "Describe your analysis request:",
            placeholder="Enter SMILES structure, UniProt IDs, or analysis description...\nExample: 'Analyze heteroatoms and similarity for SMILES CCO with UniProt P21554'",
            height=100,
            help="AI will autonomously process your request through the complete pipeline"
        )
        
        # Execution
        if st.button("🚀 Start Autonomous Analysis", type="primary", use_container_width=True) and autonomous_input:
            self._execute_autonomous_analysis(
                autonomous_input, auto_threshold, comprehensive_analysis, 
                include_visualizations, auto_download, processing_speed, result_detail
            )
    
    def _execute_manual_heteroatom_extraction(self, uniprot_input: str):
        """Execute Manual Mode heteroatom extraction with real-time PDB fetching"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        with st.spinner("🔄 Real-time PDB fetching and heteroatom extraction..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Connecting to PDB Data Bank...")
                progress_bar.progress(25)
                
                if self.agent:
                    # Use agent for real-time PDB fetching
                    results = self.agent.execute_action("heteroatom_extraction", {"uniprot_ids": uniprot_ids})
                    progress_bar.progress(100)
                    
                    if "error" not in results:
                        st.success("✅ Real-time heteroatom extraction completed!")
                        heteroatom_df = results.get("results", pd.DataFrame())
                        
                        if not heteroatom_df.empty:
                            self._display_heteroatom_results(heteroatom_df, "manual")
                        else:
                            st.warning("⚠️ No heteroatoms found for provided UniProt IDs")
                    else:
                        st.error(f"❌ Real-time extraction failed: {results['error']}")
                else:
                    st.error("❌ Agent not available for real-time PDB fetching")
                    
            except Exception as e:
                st.error(f"❌ Manual heteroatom extraction failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _execute_manual_similarity_analysis(self, target_smiles: str, radius: int, n_bits: int, 
                                          threshold: float, fp_type: str, metric: str):
        """Execute Manual Mode similarity analysis"""
        with st.spinner("🔄 Manual similarity analysis..."):
            try:
                from .similarity_analyzer import SimilarityAnalyzer
                
                analyzer = SimilarityAnalyzer(
                    radius=radius,
                    n_bits=n_bits,
                    fp_type=fp_type,
                    metric=metric
                )
                
                # Use local database for manual mode similarity comparison
                results = analyzer.analyze_similarity(
                    target_smiles=target_smiles,
                    heteroatom_df=self.local_database,
                    min_similarity=threshold
                )
                
                if not results.empty:
                    st.success("✅ Manual similarity analysis completed!")
                    self._display_similarity_results(results, target_smiles, "manual")
                else:
                    st.warning(f"⚠️ No similar compounds found with threshold {threshold}")
                    
            except Exception as e:
                st.error(f"❌ Manual similarity analysis failed: {str(e)}")
    
    def _execute_manual_combined_pipeline(self, uniprot_input: str, target_smiles: str,
                                        radius: int, n_bits: int, threshold: float, 
                                        fp_type: str, metric: str):
        """Execute Manual Mode combined pipeline"""
        uniprot_ids = [uid.strip() for uid in uniprot_input.split(',') if uid.strip()]
        
        with st.spinner("🔄 Manual combined pipeline execution..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Real-time heteroatom extraction
                status_text.text("Step 1/2: Real-time heteroatom extraction...")
                progress_bar.progress(25)
                
                heteroatom_results = pd.DataFrame()
                if self.agent:
                    hetero_response = self.agent.execute_action("heteroatom_extraction", {"uniprot_ids": uniprot_ids})
                    if "error" not in hetero_response:
                        heteroatom_results = hetero_response.get("results", pd.DataFrame())
                
                progress_bar.progress(50)
                
                # Step 2: Similarity analysis using local database
                status_text.text("Step 2/2: Molecular similarity analysis...")
                progress_bar.progress(75)
                
                from .similarity_analyzer import SimilarityAnalyzer
                analyzer = SimilarityAnalyzer(radius=radius, n_bits=n_bits, fp_type=fp_type, metric=metric)
                
                similarity_results = analyzer.analyze_similarity(
                    target_smiles=target_smiles,
                    heteroatom_df=self.local_database,
                    min_similarity=threshold
                )
                
                progress_bar.progress(100)
                
                st.success("✅ Manual combined pipeline completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if not heteroatom_results.empty:
                        st.subheader("🔬 Real-time Heteroatom Results")
                        self._display_heteroatom_results(heteroatom_results, "manual")
                
                with col2:
                    if not similarity_results.empty:
                        st.subheader("🎯 Similarity Analysis Results")
                        self._display_similarity_results(similarity_results, target_smiles, "manual")
                
            except Exception as e:
                st.error(f"❌ Manual combined pipeline failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _generate_ai_guided_response(self, user_input: str) -> str:
        """Generate AI response for guided mode with step-by-step questions"""
        user_input_lower = user_input.lower()
        
        # Extract SMILES from input
        smiles = self._extract_smiles_from_text(user_input)
        uniprot_ids = self._extract_uniprot_ids(user_input)
        
        # Initialize or update analysis state
        if "ai_smiles" not in st.session_state.current_analysis_state:
            st.session_state.current_analysis_state = {}
        
        if smiles:
            st.session_state.current_analysis_state["smiles"] = smiles
            
            response = f"""🧪 **Detected SMILES structure: `{smiles}`**

I'll guide you through the analysis using our local Excel database ({len(self.local_database)} records).

Let me ask some clarifying questions to optimize your analysis:

**❓ Question 1:** What type of analysis would you prefer?
- 🔬 **Heteroatom extraction only** (from specific UniProt IDs)
- 🎯 **Molecular similarity analysis only** (against our database)
- 🔄 **Complete pipeline** (both heteroatom + similarity)

**❓ Question 2:** What similarity threshold should I use?
- 🟢 **0.5** (loose matching - more results)
- 🟡 **0.7** (moderate matching - balanced)
- 🔴 **0.9** (strict matching - fewer, high-quality results)

**❓ Question 3:** Do you have specific UniProt IDs to focus on?
- If yes, please provide them (comma-separated)
- If no, I'll analyze all available proteins in our database

Please answer these questions and I'll proceed with your guided analysis!"""
            
            return response
        
        elif uniprot_ids:
            st.session_state.current_analysis_state["uniprot_ids"] = uniprot_ids
            
            return f"""🔍 **Detected UniProt IDs: {', '.join(uniprot_ids)}**

Great! I can extract heteroatoms from these protein structures using our local database.

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
                return self._execute_ai_guided_analysis(
                    st.session_state.current_analysis_state["smiles"],
                    analysis_type,
                    threshold,
                    uniprot_ids
                )
        
        # General welcome message
        return f"""🤖 **Welcome to AI-Powered Mode!**

I'm your analysis assistant for molecular research. I'll guide you through each step with questions to ensure optimal results.

**📊 Available Data:**
- **{len(self.local_database)} compounds** from PDB-derived Excel files
- **{self.local_database['UniProt_ID'].nunique()} UniProt proteins**
- **{self.local_database['PDB_ID'].nunique()} PDB structures**

**🎯 What I can help you with:**
- 🧬 **Molecular Similarity Analysis** - Find similar compounds
- 🔬 **Heteroatom Extraction** - Extract heteroatoms from proteins
- 📊 **Complete Pipeline Analysis** - End-to-end molecular analysis

**To get started, please provide:**
1. A SMILES structure (e.g., "Analyze this SMILES: CCO")
2. UniProt IDs (e.g., "Extract heteroatoms from P21554")
3. Or describe your analysis needs

I'll ask clarifying questions to guide you through the process!"""
    
    def _execute_ai_guided_analysis(self, smiles: str, analysis_type: str, 
                                   threshold: float, uniprot_ids: List[str] = None) -> str:
        """Execute guided analysis based on AI conversation"""
        try:
            if analysis_type == "heteroatom":
                if not uniprot_ids:
                    return "❌ Heteroatom extraction requires UniProt IDs. Please provide them."
                
                # Filter local database for specified UniProt IDs
                filtered_data = self.local_database[self.local_database['UniProt_ID'].isin(uniprot_ids)]
                if not filtered_data.empty:
                    st.subheader("🔬 AI-Guided Heteroatom Results")
                    self._display_heteroatom_results(filtered_data, "ai-guided")
                    return f"✅ **Heteroatom analysis complete!** Found {len(filtered_data)} records."
                else:
                    return f"⚠️ No data found for UniProt IDs: {', '.join(uniprot_ids)}"
            
            elif analysis_type == "similarity":
                from .similarity_analyzer import SimilarityAnalyzer
                analyzer = SimilarityAnalyzer()
                
                results = analyzer.analyze_similarity(
                    target_smiles=smiles,
                    heteroatom_df=self.local_database,
                    min_similarity=threshold
                )
                
                if not results.empty:
                    st.subheader("🎯 AI-Guided Similarity Results")
                    self._display_similarity_results(results, smiles, "ai-guided")
                    return f"✅ **Similarity analysis complete!** Found {len(results)} similar compounds."
                else:
                    return f"⚠️ No similar compounds found with threshold {threshold}"
            
            else:  # complete pipeline
                # Combined analysis
                hetero_results = pd.DataFrame()
                if uniprot_ids:
                    hetero_results = self.local_database[self.local_database['UniProt_ID'].isin(uniprot_ids)]
                
                from .similarity_analyzer import SimilarityAnalyzer
                analyzer = SimilarityAnalyzer()
                sim_results = analyzer.analyze_similarity(
                    target_smiles=smiles,
                    heteroatom_df=self.local_database,
                    min_similarity=threshold
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if not hetero_results.empty:
                        st.subheader("🔬 Heteroatom Results")
                        self._display_heteroatom_results(hetero_results, "ai-guided")
                
                with col2:
                    if not sim_results.empty:
                        st.subheader("🎯 Similarity Results")
                        self._display_similarity_results(sim_results, smiles, "ai-guided")
                
                return f"✅ **Complete pipeline analysis finished!** Heteroatoms: {len(hetero_results)}, Similar compounds: {len(sim_results)}"
                
        except Exception as e:
            return f"❌ **Analysis Error:** {str(e)}\n\nPlease try again or adjust your parameters."
    
    def _execute_autonomous_analysis(self, user_input: str, threshold: float, 
                                   comprehensive: bool, visualizations: bool,
                                   auto_download: bool, processing_speed: str, result_detail: str):
        """Execute fully autonomous analysis without user prompts"""
        with st.spinner("🤖 AI working autonomously through complete pipeline..."):
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
                
                # Step 3: Autonomous heteroatom analysis
                status_text.text("🔬 AI performing autonomous heteroatom analysis...")
                progress_bar.progress(50)
                
                heteroatom_results = pd.DataFrame()
                if uniprot_ids:
                    heteroatom_results = self.local_database[
                        self.local_database['UniProt_ID'].isin(uniprot_ids)
                    ]
                else:
                    # Use sample of database for autonomous mode
                    heteroatom_results = self.local_database.sample(min(50, len(self.local_database)))
                
                # Step 4: Autonomous similarity analysis
                status_text.text("🧪 AI performing autonomous similarity analysis...")
                progress_bar.progress(70)
                
                similarity_results = pd.DataFrame()
                if smiles:
                    from .similarity_analyzer import SimilarityAnalyzer
                    analyzer = SimilarityAnalyzer(**ai_params)
                    
                    similarity_results = analyzer.analyze_similarity(
                        target_smiles=smiles,
                        heteroatom_df=self.local_database,
                        min_similarity=ai_params["threshold"]
                    )
                
                # Step 5: Autonomous report generation
                status_text.text("📋 AI generating comprehensive autonomous report...")
                progress_bar.progress(90)
                
                # Display autonomous results
                self._display_autonomous_results(
                    smiles, uniprot_ids, heteroatom_results, similarity_results,
                    ai_params, comprehensive, visualizations, auto_download, result_detail
                )
                
                progress_bar.progress(100)
                st.success("🎉 **Autonomous Analysis Complete!** AI has processed your request end-to-end.")
                
            except Exception as e:
                st.error(f"❌ Autonomous analysis failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def _display_autonomous_results(self, smiles: str, uniprot_ids: List[str],
                                   heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                   ai_params: Dict, comprehensive: bool, visualizations: bool,
                                   auto_download: bool, result_detail: str):
        """Display comprehensive autonomous analysis results"""
        
        # Analysis overview
        st.subheader("🤖 Autonomous Analysis Overview")
        
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
        
        # Results display based on detail level
        if result_detail in ["Standard", "Detailed"]:
            if not heteroatom_results.empty:
                st.subheader("🔬 Autonomous Heteroatom Analysis")
                self._display_heteroatom_results(heteroatom_results, "autonomous")
            
            if not similarity_results.empty:
                st.subheader("🎯 Autonomous Similarity Analysis")
                self._display_similarity_results(similarity_results, smiles, "autonomous")
        
        # Comprehensive AI report
        if comprehensive:
            st.subheader("📋 AI Autonomous Comprehensive Report")
            report = self._generate_autonomous_report(
                smiles, uniprot_ids, heteroatom_results, similarity_results, ai_params
            )
            st.markdown(report)
        
        # Visualizations
        if visualizations and not similarity_results.empty:
            st.subheader("📊 Autonomous Analysis Visualizations")
            self._create_autonomous_visualizations(similarity_results, smiles)
        
        # Auto-downloads
        if auto_download:
            self._generate_autonomous_downloads(heteroatom_results, similarity_results)
    
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
    
    def _generate_autonomous_report(self, smiles: str, uniprot_ids: List[str],
                                   heteroatom_results: pd.DataFrame, similarity_results: pd.DataFrame,
                                   ai_params: Dict) -> str:
        """Generate comprehensive autonomous analysis report"""
        report = f"""
## 🤖 AI Autonomous Analysis Report

**Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 📋 Analysis Summary
- **Target SMILES:** `{smiles if smiles else 'Auto-detected from input'}`
- **UniProt IDs:** {', '.join(uniprot_ids) if uniprot_ids else 'Auto-selected from database'}
- **Data Source:** Excel-based local database ({len(self.local_database)} total records)

### ⚙️ AI-Optimized Parameters
- **Morgan Radius:** {ai_params['radius']}
- **Bit Vector Length:** {ai_params['n_bits']}
- **Fingerprint Type:** {ai_params['fp_type']}
- **Similarity Metric:** {ai_params['metric']}
- **Threshold:** {ai_params['threshold']}

### 📊 Results Overview
- **Heteroatom Records:** {len(heteroatom_results)}
- **Similar Compounds:** {len(similarity_results)}
"""
        
        if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns:
            max_sim = similarity_results['Tanimoto_Similarity'].max()
            avg_sim = similarity_results['Tanimoto_Similarity'].mean()
            report += f"""
### 🎯 Similarity Analysis
- **Highest Similarity:** {max_sim:.3f}
- **Average Similarity:** {avg_sim:.3f}
- **Confidence Level:** {'High' if avg_sim > 0.7 else 'Medium' if avg_sim > 0.5 else 'Low'}
"""
        
        report += """
### 🔮 AI Insights
- Analysis completed using local Excel database
- Future versions will integrate live PDB Data Bank connectivity
- All parameters were autonomously optimized by AI
- Results are ready for immediate download and further analysis

### ✅ Status
**AUTONOMOUS ANALYSIS COMPLETE** - No user intervention required
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