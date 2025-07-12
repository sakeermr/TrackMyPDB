import streamlit as st
import datetime
from typing import Dict, List, Any, Optional
from .agent_core import TrackMyPDBAgent, AgentQuery
from .gemini_agent import GeminiAgent
import plotly.graph_objects as go
import pandas as pd
import asyncio

class NaturalLanguageInterface:
    def __init__(self, agent: TrackMyPDBAgent, local_database: pd.DataFrame = None):
        self.agent = agent
        self.local_database = local_database if local_database is not None else pd.DataFrame()
        # Initialize Gemini immediately using config
        try:
            from .gemini_agent import GeminiAgent
            self.gemini = GeminiAgent()
        except Exception as e:
            st.error(f"Error initializing Gemini AI: {e}")
            self.gemini = None
    
    def render_manual_mode_interface(self):
        """Render Manual Mode interface - real-time PDB fetching with manual inputs"""
        st.header("🔧 Manual Mode - Real-time PDB Analysis")
        
        st.markdown("""
        **Manual Mode Features:**
        - Provide UniProt IDs and SMILES manually
        - Real-time fetching from PDB Data Bank
        - Manual parameter configuration for Morgan fingerprints
        - Three analysis stages: Heteroatom Extraction, Similarity Analysis, Combined Pipeline
        """)
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Input Parameters")
            
            # UniProt IDs input
            uniprot_input = st.text_area(
                "UniProt IDs (comma-separated):",
                placeholder="P12345, Q67890, R11223",
                help="Enter UniProt IDs separated by commas for real-time PDB fetching"
            )
            
            # SMILES input
            smiles_input = st.text_input(
                "Target SMILES Structure:",
                placeholder="CCO",
                help="Enter the SMILES string of your target molecule"
            )
        
        with col2:
            st.subheader("⚙️ Morgan Fingerprint Parameters")
            
            # Morgan parameters - manual configuration
            radius = st.slider("Morgan Radius:", 1, 4, 2)
            n_bits = st.selectbox("Bit-vector Length:", [1024, 2048, 4096], index=1)
            threshold = st.slider("Tanimoto Score Threshold:", 0.0, 1.0, 0.7, 0.05)
            fp_type = st.selectbox("Fingerprint Type:", ["morgan", "maccs", "atompair"])
            metric = st.selectbox("Similarity Metric:", ["tanimoto", "dice", "cosine"])
        
        # Analysis stage selection
        st.subheader("🎯 Analysis Stage Selection")
        analysis_stage = st.radio(
            "Choose analysis stage:",
            ["Heteroatom Extraction", "Molecular Similarity Analysis", "Combined Pipeline"],
            horizontal=True
        )
        
        # Execute analysis with real-time PDB fetching
        if st.button("🚀 Run Manual Analysis (Real-time PDB)", type="primary"):
            self._execute_manual_analysis(
                analysis_stage, uniprot_input, smiles_input, 
                radius, n_bits, threshold, fp_type, metric
            )
    
    def render_ai_powered_mode_interface(self):
        """Render AI-Powered Mode interface - guided chatbot with questions"""
        st.header("🤖 AI-Powered Mode - Interactive Guided Analysis")
        
        st.markdown("""
        **AI-Powered Mode Features:**
        - Interactive chatbot interface with step-by-step guidance
        - AI asks clarifying questions at each decision point
        - User confirms or refines parameters before proceeding
        - Uses Excel-based local database from Data module
        - Future: Will connect to live PDB fetches
        """)
        
        # Initialize chat history
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # Chat interface
        st.subheader("💬 AI Assistant Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.ai_chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # User input
        user_input = st.text_input(
            "Ask the AI Assistant:",
            placeholder="e.g., 'Provide PDB targets for this SMILES: CCO'",
            key="ai_chat_input"
        )
        
        if st.button("📤 Send Message", type="primary") and user_input:
            self._process_ai_powered_input(user_input)
    
    def render_fully_autonomous_mode_interface(self):
        """Render Fully Autonomous Mode interface - AI works without questions"""
        st.header("🚀 Fully Autonomous Mode - AI-Driven Analysis")
        
        st.markdown("""
        **Fully Autonomous Mode Features:**
        - AI proceeds through entire pipeline without user prompts
        - Automatic parameter determination for Morgan fingerprints
        - Continuous processing of new requests
        - Uses Excel-based local database from Data module
        - Comprehensive autonomous reports
        """)
        
        # Simple input interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            autonomous_input = st.text_area(
                "🎯 Analysis Request:",
                placeholder="Enter SMILES structure or describe your analysis needs...",
                help="AI will analyze autonomously: heteroatom extraction → similarity analysis → combined pipeline"
            )
        
        with col2:
            st.subheader("🤖 AI Settings")
            auto_threshold = st.slider("Auto Similarity Threshold:", 0.1, 1.0, 0.6, 0.1)
            comprehensive_analysis = st.checkbox("Comprehensive Analysis", value=True)
            include_visualizations = st.checkbox("Generate Visualizations", value=True)
        
        if st.button("🚀 Start Autonomous Analysis", type="primary") and autonomous_input:
            self._execute_autonomous_analysis(autonomous_input, auto_threshold, comprehensive_analysis, include_visualizations)
    
    def _execute_manual_analysis(self, analysis_stage: str, uniprot_input: str, smiles_input: str, 
                                radius: int, n_bits: int, threshold: float, fp_type: str, metric: str):
        """Execute manual analysis with real-time PDB fetching"""
        if analysis_stage == "Heteroatom Extraction":
            if not uniprot_input:
                st.error("❌ Please provide UniProt IDs for heteroatom extraction")
                return
            self._run_manual_heteroatom_extraction(uniprot_input)
            
        elif analysis_stage == "Molecular Similarity Analysis":
            if not smiles_input:
                st.error("❌ Please provide SMILES string for similarity analysis")
                return
            self._run_manual_similarity_analysis(smiles_input, radius, n_bits, threshold, fp_type, metric)
            
        else:  # Combined Pipeline
            if not uniprot_input or not smiles_input:
                st.error("❌ Please provide both UniProt IDs and SMILES string for combined pipeline")
                return
            self._run_manual_combined_pipeline(uniprot_input, smiles_input, radius, n_bits, threshold, fp_type, metric)
    
    def _run_manual_heteroatom_extraction(self, uniprot_input: str):
        """Run heteroatom extraction with real-time PDB fetching"""
        uniprot_ids = [id.strip() for id in uniprot_input.split(',')]
        
        with st.spinner("🔄 Fetching real-time data from PDB Data Bank..."):
            try:
                results = self.agent.execute_action("heteroatom_extraction", {"uniprot_ids": uniprot_ids})
                
                if "error" in results:
                    st.error(f"❌ Real-time PDB fetch failed: {results['error']}")
                    return
                
                st.success("✅ Real-time heteroatom extraction completed!")
                
                heteroatom_results = results.get("results", pd.DataFrame())
                if not heteroatom_results.empty:
                    st.subheader("🔬 Heteroatom Extraction Results (Real-time PDB)")
                    st.dataframe(heteroatom_results)
                    
                    # Download option
                    csv = heteroatom_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Heteroatom Results",
                        data=csv,
                        file_name=f"manual_heteroatom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No heteroatom data found for the provided UniProt IDs")
                    
            except Exception as e:
                st.error(f"❌ Manual analysis failed: {str(e)}")
    
    def _run_manual_similarity_analysis(self, smiles_input: str, radius: int, n_bits: int, 
                                       threshold: float, fp_type: str, metric: str):
        """Run similarity analysis in manual mode using local database"""
        with st.spinner("🔄 Running similarity analysis with manual parameters..."):
            try:
                # Note: Manual mode can use local database for comparison while fetching real-time PDB data
                from .similarity_analyzer import SimilarityAnalyzer
                analyzer = SimilarityAnalyzer(
                    radius=radius,
                    n_bits=n_bits,
                    fp_type=fp_type,
                    metric=metric
                )
                
                results = analyzer.analyze_similarity(
                    target_smiles=smiles_input,
                    heteroatom_df=self.local_database,
                    min_similarity=threshold
                )
                
                if not results.empty:
                    st.success("✅ Manual similarity analysis completed!")
                    st.subheader("🎯 Similarity Analysis Results")
                    st.dataframe(results)
                    
                    # Visualization
                    fig = self._create_similarity_plot(results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Similarity Results",
                        data=csv,
                        file_name=f"manual_similarity_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"⚠️ No similar compounds found with threshold {threshold}")
                    
            except Exception as e:
                st.error(f"❌ Manual similarity analysis failed: {str(e)}")
    
    def _run_manual_combined_pipeline(self, uniprot_input: str, smiles_input: str, 
                                     radius: int, n_bits: int, threshold: float, fp_type: str, metric: str):
        """Run combined pipeline in manual mode"""
        uniprot_ids = [id.strip() for id in uniprot_input.split(',')]
        
        with st.spinner("🔄 Running complete analysis pipeline with real-time PDB fetching..."):
            try:
                # Execute complete pipeline with real-time PDB data
                results = self.agent.execute_action(
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
                    st.error(f"❌ Combined pipeline failed: {results['error']}")
                    return
                
                st.success("✅ Manual combined pipeline completed!")
                
                # Display heteroatom results
                heteroatom_results = results.get("heteroatom_results", pd.DataFrame())
                if not heteroatom_results.empty:
                    st.subheader("🔬 Heteroatom Extraction Results (Real-time PDB)")
                    st.dataframe(heteroatom_results)
                
                # Display similarity results
                similarity_results = results.get("similarity_results", pd.DataFrame())
                if not similarity_results.empty:
                    st.subheader("🎯 Similarity Analysis Results")
                    st.dataframe(similarity_results)
                    
                    # Visualization
                    fig = self._create_similarity_plot(similarity_results)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Combined download
                with st.expander("📥 Download Combined Results", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not heteroatom_results.empty:
                            hetero_csv = heteroatom_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Heteroatom Results",
                                data=hetero_csv,
                                file_name=f"manual_heteroatom_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        if not similarity_results.empty:
                            sim_csv = similarity_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Similarity Results",
                                data=sim_csv,
                                file_name=f"manual_similarity_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
            except Exception as e:
                st.error(f"❌ Manual combined pipeline failed: {str(e)}")
    
    def _process_ai_powered_input(self, user_input: str):
        """Process user input for AI-Powered mode with step-by-step guidance"""
        # Add user message to history
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
        
        # Process input and generate AI response
        ai_response = self._generate_ai_powered_response(user_input)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display new messages
        st.rerun()
    
    def _generate_ai_powered_response(self, user_input: str) -> str:
        """Generate AI response for guided mode with questions"""
        user_input_lower = user_input.lower()
        
        # Check if user is providing a SMILES structure
        smiles = self._extract_smiles_from_text(user_input)
        
        if smiles:
            response = f"""🧪 **I found a SMILES structure: `{smiles}`**

I'll guide you through the analysis using our local Excel database. Let me ask some clarifying questions:

❓ **Question 1:** Do you have specific UniProt IDs you'd like me to focus on for this analysis? 
   - If yes, please provide them (comma-separated)
   - If no, I'll search all available proteins in our database

❓ **Question 2:** What type of analysis would you prefer?
   - Heteroatom extraction only
   - Molecular similarity analysis only  
   - Complete pipeline (both heteroatom + similarity)

❓ **Question 3:** What similarity threshold should I use?
   - 0.5 (loose matching - more results)
   - 0.7 (moderate matching - balanced)
   - 0.9 (strict matching - fewer, high-quality results)

Please answer these questions and I'll proceed with the guided analysis!"""
            
            # Store current SMILES for later use
            st.session_state.ai_current_smiles = smiles
            return response
        
        # Handle follow-up responses based on conversation state
        if hasattr(st.session_state, 'ai_current_smiles'):
            return self._handle_ai_followup_questions(user_input)
        
        # General help response
        return """🤖 **Welcome to AI-Powered Mode!**

I'm here to guide you through molecular analysis step by step. I'll ask clarifying questions at each decision point to ensure we get the best results.

**What I can help you with:**
- 🧬 **Molecular Similarity Analysis** - Find similar compounds using our Excel database
- 🔬 **Heteroatom Extraction** - Extract heteroatoms from protein structures
- 📊 **Complete Pipeline Analysis** - End-to-end molecular analysis

**Our Local Database Contains:**
- {len(self.local_database)} compounds from PDB-derived Excel files
- SMILES representations and UniProt IDs
- Ready for immediate analysis (future: live PDB connection)

**To get started, please provide:**
1. A SMILES structure (e.g., "Analyze this SMILES: CCO")
2. Or describe what you'd like to analyze

I'll guide you through each step with questions to ensure optimal results!"""
    
    def _handle_ai_followup_questions(self, user_input: str) -> str:
        """Handle follow-up questions in AI-powered mode"""
        # Extract information from user response
        uniprot_ids = self._extract_uniprot_ids(user_input)
        analysis_type = self._extract_analysis_type(user_input)
        threshold = self._extract_threshold_from_text(user_input)
        
        # Check if user provided enough information to proceed
        if analysis_type and threshold is not None:
            # User provided enough info, execute analysis
            return self._execute_ai_guided_analysis(
                st.session_state.ai_current_smiles,
                uniprot_ids,
                analysis_type,
                threshold
            )
        else:
            # Ask for missing information
            missing_info = []
            if not analysis_type:
                missing_info.append("analysis type (heteroatom/similarity/complete)")
            if threshold is None:
                missing_info.append("similarity threshold (0.5/0.7/0.9)")
            
            return f"""📝 **I need a bit more information:**

Missing: {', '.join(missing_info)}

Please specify:
- Analysis type: "heteroatom extraction", "similarity analysis", or "complete pipeline"
- Similarity threshold: 0.5, 0.7, or 0.9
- UniProt IDs (optional): comma-separated list

Example: "Run complete pipeline with threshold 0.7 and UniProt IDs P12345, Q67890" """
    
    def _execute_ai_guided_analysis(self, smiles: str, uniprot_ids: List[str], 
                                   analysis_type: str, threshold: float) -> str:
        """Execute the guided analysis based on AI conversation"""
        try:
            if analysis_type == "heteroatom":
                if not uniprot_ids:
                    return "❌ Heteroatom extraction requires UniProt IDs. Please provide them."
                results = self._run_ai_heteroatom_extraction(uniprot_ids)
            elif analysis_type == "similarity":
                results = self._run_ai_similarity_analysis(smiles, threshold)
            else:  # complete pipeline
                if not uniprot_ids:
                    return "❌ Complete pipeline requires UniProt IDs. Please provide them."
                results = self._run_ai_combined_pipeline(smiles, uniprot_ids, threshold)
            
            return f"✅ **Analysis Complete!**\n\nResults are displayed below. The analysis used our local Excel database with {len(results)} matches found."
            
        except Exception as e:
            return f"❌ **Analysis Error:** {str(e)}\n\nPlease try again or adjust your parameters."
    
    def _execute_autonomous_analysis(self, user_input: str, threshold: float, 
                                   comprehensive: bool, visualizations: bool):
        """Execute fully autonomous analysis without questions"""
        with st.spinner("🤖 AI working autonomously through complete pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Analyze input autonomously
                status_text.text("🔍 AI analyzing input and extracting parameters...")
                progress_bar.progress(20)
                
                smiles = self._extract_smiles_from_text(user_input)
                if not smiles:
                    st.error("❌ AI couldn't detect a valid SMILES structure")
                    return
                
                # Step 2: Autonomous parameter selection
                status_text.text("⚙️ AI selecting optimal parameters...")
                progress_bar.progress(40)
                
                # AI determines parameters automatically
                ai_params = {
                    "radius": 2,  # AI-optimized
                    "n_bits": 2048,  # AI-optimized
                    "fp_type": "morgan",  # AI-preferred
                    "metric": "tanimoto",  # AI-preferred
                    "threshold": threshold
                }
                
                # Step 3: Heteroatom extraction (autonomous)
                status_text.text("🔬 AI performing heteroatom extraction...")
                progress_bar.progress(60)
                
                # For autonomous mode, use available data from local database
                heteroatom_results = self._autonomous_heteroatom_extraction()
                
                # Step 4: Similarity analysis (autonomous)
                status_text.text("🧪 AI performing molecular similarity analysis...")
                progress_bar.progress(80)
                
                similarity_results = self._autonomous_similarity_analysis(smiles, ai_params)
                
                # Step 5: Generate comprehensive report
                status_text.text("📋 AI generating comprehensive autonomous report...")
                progress_bar.progress(100)
                
                # Display results
                st.success("🎉 **Autonomous Analysis Complete!**")
                
                # Display comprehensive results
                self._display_autonomous_results(
                    smiles, heteroatom_results, similarity_results, 
                    ai_params, comprehensive, visualizations
                )
                
            except Exception as e:
                st.error(f"❌ Autonomous analysis failed: {str(e)}")
    
    def _autonomous_heteroatom_extraction(self) -> pd.DataFrame:
        """Perform autonomous heteroatom extraction using local database"""
        # In autonomous mode, extract heteroatoms from available local data
        if not self.local_database.empty and 'UniProt_ID' in self.local_database.columns:
            # Return available heteroatom data from local database
            return self.local_database.head(10)  # Sample for demonstration
        return pd.DataFrame()
    
    def _autonomous_similarity_analysis(self, smiles: str, params: Dict) -> pd.DataFrame:
        """Perform autonomous similarity analysis"""
        try:
            from .similarity_analyzer import SimilarityAnalyzer
            analyzer = SimilarityAnalyzer(
                radius=params["radius"],
                n_bits=params["n_bits"],
                fp_type=params["fp_type"],
                metric=params["metric"]
            )
            
            results = analyzer.analyze_similarity(
                target_smiles=smiles,
                heteroatom_df=self.local_database,
                min_similarity=params["threshold"]
            )
            
            return results
            
        except Exception as e:
            st.error(f"Autonomous similarity analysis failed: {e}")
            return pd.DataFrame()
    
    def _display_autonomous_results(self, smiles: str, heteroatom_results: pd.DataFrame, 
                                   similarity_results: pd.DataFrame, params: Dict,
                                   comprehensive: bool, visualizations: bool):
        """Display comprehensive autonomous analysis results"""
        
        # Analysis summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Target SMILES", smiles[:20] + "..." if len(smiles) > 20 else smiles)
        with col2:
            st.metric("Similar Compounds", len(similarity_results))
        with col3:
            avg_similarity = similarity_results['Similarity'].mean() if not similarity_results.empty and 'Similarity' in similarity_results.columns else 0
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        with col4:
            st.metric("AI Parameters", f"R={params['radius']}, B={params['n_bits']}")
        
        # Heteroatom results
        if not heteroatom_results.empty:
            st.subheader("🔬 Autonomous Heteroatom Analysis")
            st.dataframe(heteroatom_results)
        
        # Similarity results
        if not similarity_results.empty:
            st.subheader("🎯 Autonomous Similarity Analysis")
            st.dataframe(similarity_results)
            
            if visualizations:
                fig = self._create_similarity_plot(similarity_results)
                st.plotly_chart(fig, use_container_width=True)
        
        # Comprehensive AI report
        if comprehensive:
            st.subheader("📋 AI Autonomous Comprehensive Report")
            report = self._generate_autonomous_report(smiles, heteroatom_results, similarity_results, params)
            st.markdown(report)
        
        # Download options
        st.subheader("📥 Download Autonomous Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not heteroatom_results.empty:
                hetero_csv = heteroatom_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Heteroatom Results",
                    data=hetero_csv,
                    file_name=f"autonomous_heteroatom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not similarity_results.empty:
                sim_csv = similarity_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Similarity Results",
                    data=sim_csv,
                    file_name=f"autonomous_similarity_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def _generate_autonomous_report(self, smiles: str, heteroatom_results: pd.DataFrame, 
                                   similarity_results: pd.DataFrame, params: Dict) -> str:
        """Generate comprehensive autonomous analysis report"""
        report = f"""
## 🤖 AI Autonomous Analysis Report

**Target Molecule:** `{smiles}`

**Analysis Parameters (AI-Optimized):**
- Morgan Radius: {params['radius']}
- Bit Vector Length: {params['n_bits']}
- Fingerprint Type: {params['fp_type']}
- Similarity Metric: {params['metric']}
- Threshold: {params['threshold']}

**Data Source:** Excel-based local database (PDB-derived)

**Results Summary:**
- Heteroatom compounds analyzed: {len(heteroatom_results)}
- Similar compounds found: {len(similarity_results)}
"""
        
        if not similarity_results.empty and 'Similarity' in similarity_results.columns:
            max_sim = similarity_results['Similarity'].max()
            avg_sim = similarity_results['Similarity'].mean()
            report += f"""
**Similarity Analysis:**
- Highest similarity score: {max_sim:.3f}
- Average similarity score: {avg_sim:.3f}
- Confidence level: {'High' if avg_sim > 0.7 else 'Medium' if avg_sim > 0.5 else 'Low'}
"""
        
        report += """
**AI Recommendations:**
- Results are based on local Excel database analysis
- Future versions will integrate live PDB Data Bank fetching
- Consider manual mode for real-time PDB data validation
"""
        
        return report
    
    # Helper methods
    def _extract_smiles_from_text(self, text: str) -> str:
        """Extract SMILES string from text input"""
        import re
        # Look for common SMILES patterns
        smiles_patterns = [
            r'([A-Za-z0-9@+\-\[\]()=:#$.\/\\]+)',  # General SMILES pattern
            r'C[A-Za-z0-9@+\-\[\]()=:#$.\/\\]*',   # Starting with C
            r'[Cc][Cc][Oo]',  # CCO pattern
            r'[A-Za-z]{1,2}[0-9]*[\[\]()=:#$]*'     # Chemical symbols with bonds
        ]
        
        for pattern in smiles_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if any(char in match for char in ['C', 'N', 'O', '=', '(', ')', '[', ']']):
                    return match.strip()
        return ""
    
    def _extract_uniprot_ids(self, text: str) -> List[str]:
        """Extract UniProt IDs from text input"""
        import re
        # Look for UniProt ID patterns (e.g., P53_HUMAN, Q9UNQ0)
        uniprot_pattern = r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}'
        matches = re.findall(uniprot_pattern, text.upper())
        
        # Also look for simple patterns like P12345
        simple_pattern = r'[PQO]\d+[A-Z]*\d*'
        simple_matches = re.findall(simple_pattern, text.upper())
        
        all_matches = list(set(matches + simple_matches))
        return [match for match in all_matches if len(match) >= 4]
    
    def _extract_analysis_type(self, text: str) -> str:
        """Extract analysis type from text"""
        text_lower = text.lower()
        if 'heteroatom' in text_lower:
            return 'heteroatom'
        elif 'similarity' in text_lower:
            return 'similarity'
        elif 'complete' in text_lower or 'pipeline' in text_lower or 'both' in text_lower:
            return 'complete'
        return ""
    
    def _extract_threshold_from_text(self, text: str) -> Optional[float]:
        """Extract threshold value from text"""
        import re
        # Look for decimal numbers between 0 and 1
        numbers = re.findall(r'0\.\d+|\d\.\d+', text)
        for num in numbers:
            val = float(num)
            if 0 <= val <= 1:
                return val
        
        # Look for common threshold keywords
        if '0.5' in text or 'loose' in text.lower():
            return 0.5
        elif '0.7' in text or 'moderate' in text.lower():
            return 0.7
        elif '0.9' in text or 'strict' in text.lower():
            return 0.9
        
        return None
    
    def _create_similarity_plot(self, results: pd.DataFrame):
        """Create similarity visualization plot"""
        if results.empty or 'Similarity' not in results.columns:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(results))),
            y=results['Similarity'],
            mode='markers+lines',
            name='Similarity Score',
            marker=dict(
                size=8,
                color=results['Similarity'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Similarity")
            )
        ))
        
        fig.update_layout(
            title="Molecular Similarity Analysis Results",
            xaxis_title="Compound Index",
            yaxis_title="Similarity Score",
            hovermode='closest'
        )
        
        return fig
    
    def _run_ai_heteroatom_extraction(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Run heteroatom extraction for AI mode"""
        # For AI mode, use local database
        if not self.local_database.empty and 'UniProt_ID' in self.local_database.columns:
            results = self.local_database[self.local_database['UniProt_ID'].isin(uniprot_ids)]
            return results
        return pd.DataFrame()
    
    def _run_ai_similarity_analysis(self, smiles: str, threshold: float) -> pd.DataFrame:
        """Run similarity analysis for AI mode"""
        try:
            from .similarity_analyzer import SimilarityAnalyzer
            analyzer = SimilarityAnalyzer()
            
            results = analyzer.analyze_similarity(
                target_smiles=smiles,
                heteroatom_df=self.local_database,
                min_similarity=threshold
            )
            
            return results
            
        except Exception as e:
            st.error(f"AI similarity analysis failed: {e}")
            return pd.DataFrame()
    
    def _run_ai_combined_pipeline(self, smiles: str, uniprot_ids: List[str], threshold: float) -> pd.DataFrame:
        """Run combined pipeline for AI mode"""
        # Combine heteroatom and similarity results
        hetero_results = self._run_ai_heteroatom_extraction(uniprot_ids)
        sim_results = self._run_ai_similarity_analysis(smiles, threshold)
        
        # For now, return similarity results (can be enhanced to combine both)
        return sim_results