import streamlit as st
import datetime
from typing import Dict, List, Any
from .agent_core import TrackMyPDBAgent, AgentQuery
import plotly.graph_objects as go
import pandas as pd

class NaturalLanguageInterface:
    def __init__(self, agent: TrackMyPDBAgent):
        self.agent = agent
    
    def render_chat_interface(self):
        """Render the natural language chat interface"""
        st.subheader("🧬 Ask TrackMyPDB")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.text_input(
            "Ask about protein-ligand analysis:",
            placeholder="e.g., 'Extract heteroatoms from P53 and analyze similarity'"
        )
        
        if st.button("Send") and user_input:
            # Process user input synchronously since we're in Streamlit
            self._process_user_input_sync(user_input)
        
        # Display chat history
        self._display_chat_history()
    
    def _process_user_input_sync(self, user_input: str):
        """Process user input synchronously for Streamlit compatibility"""
        # Create query
        query = AgentQuery(
            text=user_input,
            timestamp=datetime.datetime.now(),
            query_type="complete_pipeline"  # Default to complete pipeline for NL queries
        )
        
        # Add to chat history
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
            
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_input,
            "timestamp": query.timestamp.isoformat()
        })
        
        # Extract potential parameters from user input
        params = self._extract_parameters(user_input)
        
        # Process with agent
        with st.spinner("🧠 Analyzing your request..."):
            response = self.agent.execute_action(
                action_name=params["action"],
                parameters=params["parameters"]
            )
        
        # Format response
        response_text = self._format_response(response)
        
        # Add agent response to chat history
        st.session_state.chat_history.append({
            "type": "agent",
            "content": response_text,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract action and parameters from natural language input"""
        # Simple rule-based parameter extraction
        text = text.lower()
        params = {"parameters": {}}
        
        if "extract" in text and "heteroatom" in text:
            params["action"] = "extract_heteroatoms"
            # Extract UniProt IDs - look for common patterns
            uniprot_ids = []
            words = text.split()
            for word in words:
                # Look for typical UniProt ID patterns (e.g., P53, Q9UNQ0)
                if (len(word) >= 3 and 
                    (word[0] in ['p', 'q', 'o'] and 
                     any(c.isdigit() for c in word))):
                    uniprot_ids.append(word.upper())
            params["parameters"]["uniprot_ids"] = uniprot_ids
            
        elif "similarity" in text or "similar" in text:
            params["action"] = "analyze_similarity"
            # Look for SMILES strings (typically containing parentheses and equals signs)
            words = text.split()
            for word in words:
                if "(" in word and ")" in word or "=" in word:
                    params["parameters"]["smiles"] = word
                    break
            params["parameters"]["threshold"] = 0.7  # Default threshold
            
        else:
            # Default to complete pipeline
            params["action"] = "complete_pipeline"
            params["parameters"] = {
                "uniprot_ids": [],
                "smiles": "",
                "threshold": 0.7
            }
        
        return params
    
    def _format_response(self, response: Dict[str, Any]) -> str:
        """Format the agent's response for display"""
        if "error" in response:
            return f"❌ Error: {response['error']}"
        
        if "results" in response:
            if isinstance(response["results"], pd.DataFrame):
                return "✅ Analysis complete! Results are shown below."
            return f"✅ Analysis complete! Found {len(response['results'])} results."
            
        if "heteroatom_results" in response and "similarity_results" in response:
            return "✅ Complete analysis finished! Both heteroatom and similarity results are shown below."
            
        return "✅ Analysis complete!"
    
    def _display_chat_history(self):
        """Display chat history with rich formatting"""
        if hasattr(st.session_state, 'chat_history'):
            for message in st.session_state.chat_history:
                if message["type"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")
    
    def _display_results(self, results: Dict[str, Any], action_type: str):
        """Display analysis results with rich visualizations"""
        if isinstance(results, dict):
            if "heteroatom_results" in results:
                st.subheader("📊 Heteroatom Analysis Results")
                st.dataframe(results["heteroatom_results"])
                
            if "similarity_results" in results:
                st.subheader("🧪 Similarity Analysis Results")
                st.dataframe(results["similarity_results"])