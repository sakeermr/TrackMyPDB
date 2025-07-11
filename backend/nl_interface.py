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
            self._process_user_input(user_input)
        
        # Display chat history
        self._display_chat_history()
    
    async def _process_user_input(self, user_input: str):
        """Process user input and generate agent response"""
        # Create query
        query = AgentQuery(
            text=user_input,
            timestamp=datetime.datetime.now(),
            query_type="heteroatom_analysis"  # Default type
        )
        
        # Add to chat history
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
            
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_input,
            "timestamp": query.timestamp.isoformat()
        })
        
        # Process with agent
        with st.spinner("🧠 Analyzing your request..."):
            response = await self.agent.process_query(query)
        
        # Add agent response to chat history
        st.session_state.chat_history.append({
            "type": "agent",
            "content": str(response),
            "timestamp": datetime.datetime.now().isoformat()
        })
    
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
        if isinstance(results, dict) and "heteroatom_results" in results:
            df = pd.DataFrame(results["heteroatom_results"])
            if not df.empty:
                st.dataframe(df)