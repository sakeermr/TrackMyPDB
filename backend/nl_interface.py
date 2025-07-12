import streamlit as st
import datetime
from typing import Dict, List, Any
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
    
    def render_chat_interface(self, mode: str = "ai_powered"):
        """Render the natural language chat interface for AI modes"""
        if mode == "ai_powered":
            st.subheader("🤖 AI-Powered Analysis Mode")
            st.info("💡 I'll ask you questions to guide the analysis step by step.")
        else:  # fully_autonomous
            st.subheader("🚀 Fully Autonomous Analysis Mode")
            st.info("🤖 I'll analyze autonomously without asking questions.")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.text_input(
            "Enter your analysis request:",
            placeholder="e.g., 'Provide PDB targets for this SMILES: CCO' or 'Find similar compounds to aspirin'"
        )
        
        if st.button("Send") and user_input:
            self._process_ai_input(user_input, mode)
        
        # Display chat history
        self._display_chat_history()
    
    def _process_ai_input(self, user_input: str, mode: str):
        """Process user input for AI modes using local database"""
        # Add to chat history
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Process based on mode
        if mode == "ai_powered":
            self._process_ai_powered_mode(user_input)
        else:  # fully_autonomous
            self._process_autonomous_mode(user_input)
    
    def _process_ai_powered_mode(self, user_input: str):
        """Process input for AI-Powered mode (asks questions)"""
        # Extract SMILES from input
        smiles = self._extract_smiles_from_text(user_input)
        
        if smiles:
            # Ask user for confirmation and parameters
            response = f"🧪 I found a SMILES structure: `{smiles}`\n\n"
            response += "I'll search our local database for similar compounds. "
            
            # Ask for UniProt IDs if mentioned
            if "uniprot" in user_input.lower() or "protein" in user_input.lower():
                response += "\n\n❓ **Question:** Do you have any specific UniProt IDs you'd like me to focus on for this analysis?"
                st.session_state.awaiting_uniprot_input = True
                st.session_state.current_smiles = smiles
            else:
                response += "\n\n❓ **Question:** What similarity threshold would you like me to use? (0.5 = loose, 0.7 = moderate, 0.9 = strict)"
                st.session_state.awaiting_threshold_input = True
                st.session_state.current_smiles = smiles
            
            self._add_agent_response(response)
            return
        
        # Handle follow-up responses
        if hasattr(st.session_state, 'awaiting_uniprot_input') and st.session_state.awaiting_uniprot_input:
            uniprot_ids = self._extract_uniprot_ids(user_input)
            st.session_state.current_uniprot_ids = uniprot_ids
            st.session_state.awaiting_uniprot_input = False
            
            response = f"✅ Got it! UniProt IDs: {', '.join(uniprot_ids) if uniprot_ids else 'None specified'}\n\n"
            response += "❓ **Question:** What similarity threshold should I use? (0.5 = loose, 0.7 = moderate, 0.9 = strict)"
            st.session_state.awaiting_threshold_input = True
            self._add_agent_response(response)
            return
        
        if hasattr(st.session_state, 'awaiting_threshold_input') and st.session_state.awaiting_threshold_input:
            threshold = self._extract_threshold(user_input)
            st.session_state.awaiting_threshold_input = False
            
            response = f"✅ Using similarity threshold: {threshold}\n\n"
            response += "🔍 **Confirming Analysis Parameters:**\n"
            response += f"- SMILES: `{st.session_state.current_smiles}`\n"
            response += f"- UniProt IDs: {getattr(st.session_state, 'current_uniprot_ids', 'None')}\n"
            response += f"- Threshold: {threshold}\n\n"
            response += "❓ **Question:** Should I proceed with this analysis? (yes/no)"
            st.session_state.awaiting_confirmation = True
            st.session_state.current_threshold = threshold
            self._add_agent_response(response)
            return
        
        if hasattr(st.session_state, 'awaiting_confirmation') and st.session_state.awaiting_confirmation:
            if "yes" in user_input.lower() or "proceed" in user_input.lower():
                st.session_state.awaiting_confirmation = False
                self._execute_ai_analysis()
            else:
                st.session_state.awaiting_confirmation = False
                self._add_agent_response("❌ Analysis cancelled. Please provide new parameters.")
            return
        
        # If no specific pattern, provide general help
        response = """🤖 I can help you with molecular analysis using our local database!

**What I can do:**
- Find PDB targets for SMILES structures
- Analyze molecular similarity
- Extract heteroatom information

**Examples:**
- "Provide PDB targets for this SMILES: CCO"
- "Find compounds similar to aspirin"
- "Analyze similarity for SMILES: C1=CC=CC=C1"

Please provide a SMILES structure to get started!"""
        
        self._add_agent_response(response)
    
    def _process_autonomous_mode(self, user_input: str):
        """Process input for Fully Autonomous mode (no questions)"""
        response = "🚀 **Autonomous Analysis Started**\n\n"
        
        # Extract SMILES
        smiles = self._extract_smiles_from_text(user_input)
        
        if not smiles:
            # Try to identify compound by name and find SMILES in database
            compound_name = self._extract_compound_name(user_input)
            if compound_name:
                smiles = self._find_smiles_by_name(compound_name)
                if smiles:
                    response += f"🔍 Found SMILES for {compound_name}: `{smiles}`\n"
                else:
                    response += f"❌ Could not find SMILES for compound: {compound_name}"
                    self._add_agent_response(response)
                    return
            else:
                response += "❌ No SMILES structure or recognizable compound found in input."
                self._add_agent_response(response)
                return
        
        # Autonomous parameter selection
        threshold = 0.7  # Default moderate threshold
        uniprot_ids = self._extract_uniprot_ids(user_input)
        
        response += f"⚙️ **Autonomous Parameter Selection:**\n"
        response += f"- SMILES: `{smiles}`\n"
        response += f"- Similarity Threshold: {threshold} (moderate)\n"
        response += f"- UniProt Filter: {'Yes' if uniprot_ids else 'No'}\n\n"
        response += "🔄 **Executing Analysis...**"
        
        self._add_agent_response(response)
        
        # Store parameters and execute
        st.session_state.current_smiles = smiles
        st.session_state.current_threshold = threshold
        st.session_state.current_uniprot_ids = uniprot_ids if uniprot_ids else []
        
        self._execute_ai_analysis(autonomous=True)
    
    def _execute_ai_analysis(self, autonomous: bool = False):
        """Execute the analysis with current parameters"""
        with st.spinner("🔬 Analyzing molecular data..."):
            try:
                # Search local database
                results = self._search_local_database(
                    st.session_state.current_smiles,
                    st.session_state.current_threshold,
                    getattr(st.session_state, 'current_uniprot_ids', [])
                )
                
                if not results.empty:
                    response = f"✅ **Analysis Complete!**\n\n"
                    response += f"📊 Found {len(results)} similar compounds in local database\n"
                    response += f"🎯 Top similarity score: {results['Similarity'].max():.3f}\n"
                    response += f"📈 Average similarity: {results['Similarity'].mean():.3f}\n\n"
                    
                    if autonomous:
                        response += "🤖 **Autonomous Insights:**\n"
                        response += self._generate_autonomous_insights(results)
                    else:
                        response += "💡 Results are displayed below. Would you like me to provide detailed insights?"
                    
                    # Store results for display
                    st.session_state.current_results = results
                    
                else:
                    response = f"❌ No similar compounds found with threshold {st.session_state.current_threshold}"
                    response += "\n💡 Try lowering the similarity threshold or check the SMILES structure."
                
                self._add_agent_response(response)
                
                # Display results if found
                if not results.empty:
                    st.subheader("📊 Analysis Results")
                    st.dataframe(results)
                    
                    # Generate download link
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results",
                        data=csv,
                        file_name=f"similarity_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                error_response = f"❌ **Analysis Error:** {str(e)}"
                self._add_agent_response(error_response)
    
    def _search_local_database(self, smiles: str, threshold: float, uniprot_filter: List[str] = None) -> pd.DataFrame:
        """Search local database for similar compounds"""
        if self.local_database.empty:
            return pd.DataFrame()
        
        try:
            from .similarity_analyzer import SimilarityAnalyzer
            analyzer = SimilarityAnalyzer()
            
            # Filter by UniProt IDs if specified
            search_df = self.local_database
            if uniprot_filter:
                search_df = search_df[search_df['UniProt_ID'].isin(uniprot_filter)]
            
            results = analyzer.analyze_similarity(
                target_smiles=smiles,
                heteroatom_df=search_df,
                min_similarity=threshold
            )
            
            return results
            
        except Exception as e:
            st.error(f"Database search failed: {e}")
            return pd.DataFrame()
    
    def _extract_smiles_from_text(self, text: str) -> str:
        """Extract SMILES string from text input"""
        # Simple heuristic: look for text within parentheses or common SMILES patterns
        import re
        match = re.search(r'([Cc]1=|[Cc]1-|[Cc]=|[Cc]#|[Cc]\/|[Cc]\\\\|[Cc]\s*=\s*|[Cc]\s*#\s*|[Cc]\/\s*|[Cc]\\\\\s*)', text)
        if match:
            start = text.find(match.group(0))
            end = start + len(match.group(0))
            return text[start:end].strip()
        return ""
    
    def _extract_uniprot_ids(self, text: str) -> List[str]:
        """Extract UniProt IDs from text input"""
        text = text.lower()
        uniprot_ids = []
        
        # Look for typical UniProt ID patterns (e.g., P53, Q9UNQ0)
        words = text.split()
        for word in words:
            if (len(word) >= 3 and 
                (word[0] in ['p', 'q', 'o'] and 
                 any(c.isdigit() for c in word))):
                uniprot_ids.append(word.upper())
        
        return uniprot_ids
    
    def _extract_threshold(self, text: str) -> float:
        """Extract similarity threshold from text input"""
        text = text.lower()
        
        for word in text.split():
            try:
                value = float(word)
                if 0 <= value <= 1:
                    return value
            except ValueError:
                continue
        
        return 0.7  # Default threshold
    
    def _extract_compound_name(self, text: str) -> str:
        """Extract compound name from text input (heuristic approach)"""
        # Simple heuristic: take the first part of the text before any special characters
        import re
        match = re.match(r'^[^\W\d_]+', text)
        if match:
            return match.group(0)
        return ""
    
    def _find_smiles_by_name(self, compound_name: str) -> str:
        """Find SMILES string by compound name using local database"""
        if self.local_database.empty:
            return ""
        
        # Heuristic search: look for exact or partial matches in the database
        matches = self.local_database[self.local_database['Compound_Name'].str.contains(compound_name, case=False, na=False)]
        
        if not matches.empty:
            # Return the first matching SMILES
            return matches.iloc[0]['SMILES']
        
        return ""
    
    def _add_agent_response(self, response: str):
        """Add agent response to chat history and display"""
        st.session_state.chat_history.append({
            "type": "agent",
            "content": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Display the new response
        st.markdown(f"**🤖 Assistant:** {response}")
    
    def _generate_autonomous_insights(self, results: pd.DataFrame) -> str:
        """Generate insights based on analysis results (autonomous mode)"""
        if results.empty:
            return "No insights available - no results found."
        
        # Example insights: distribution of similarity scores, common features
        insights = []
        
        # Distribution of similarity scores
        if 'Similarity' in results.columns:
            mean_similarity = results['Similarity'].mean()
            max_similarity = results['Similarity'].max()
            min_similarity = results['Similarity'].min()
            
            insights.append(f"🔍 **Similarity Score Distribution:**")
            insights.append(f"- Mean: {mean_similarity:.3f}")
            insights.append(f"- Max: {max_similarity:.3f}")
            insights.append(f"- Min: {min_similarity:.3f}")
        
        # Common features in similar compounds (e.g., common substructures)
        # For simplicity, let's say we define common features as those present in >50% of the top 10 results
        if len(results) > 10:
            top_results = results.nlargest(10, 'Similarity')
        else:
            top_results = results
        
        # Heuristic: look for common substructure patterns in SMILES
        common_substructures = []
        for smiles in top_results['SMILES']:
            # Extract substructures (e.g., ethylene from C=C)
            substructures = self._extract_substructures_from_smiles(smiles)
            common_substructures.extend(substructures)
        
        # Find the most common substructures
        from collections import Counter
        if common_substructures:
            substructure_counts = Counter(common_substructures)
            most_common_substructures = substructure_counts.most_common(3)
            insights.append("🔗 **Common Substructures in Similar Compounds:**")
            for substructure, count in most_common_substructures:
                insights.append(f"- `{substructure}` (found in {count} compounds)")
        
        return "\n".join(insights)
    
    def _extract_substructures_from_smiles(self, smiles: str) -> List[str]:
        """Extract substructures from SMILES string (heuristic approach)"""
        # Simple heuristic: split by common separators and filter fragments
        fragments = [frag.strip() for frag in smiles.replace('=', '/').replace('#', '/').split('/')]
        filtered_fragments = [frag for frag in fragments if len(frag) > 1]
        return filtered_fragments