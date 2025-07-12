import streamlit as st
import datetime
from typing import Dict, List, Any
from .agent_core import TrackMyPDBAgent, AgentQuery
from .gemini_agent import GeminiAgent
import plotly.graph_objects as go
import pandas as pd

class NaturalLanguageInterface:
    def __init__(self, agent: TrackMyPDBAgent):
        self.agent = agent
        # Initialize Gemini immediately using config
        try:
            from .gemini_agent import GeminiAgent
            self.gemini = GeminiAgent()
        except Exception as e:
            st.error(f"Error initializing Gemini AI: {e}")
            self.gemini = None
    
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
    
    async def _process_user_input_sync(self, user_input: str):
        """Process user input synchronously for Streamlit compatibility"""
        # Add to chat history
        if not hasattr(st.session_state, 'chat_history'):
            st.session_state.chat_history = []
        
        # Check if this is a continuation query
        if user_input.lower().strip() in ["continue", "continue to iterate", "continue iteration"]:
            if hasattr(st.session_state, 'last_results'):
                should_continue = self.gemini.should_continue_iteration(st.session_state.last_results)
                if should_continue:
                    # Perform another iteration with previous parameters
                    with st.spinner("🔄 Continuing analysis..."):
                        response = self.agent.execute_action(
                            action_name=st.session_state.last_action,
                            parameters=st.session_state.last_parameters
                        )
                        st.session_state.last_results = response
                        
                        response_text = "✅ Iteration complete!\n\n"
                        if self.gemini:
                            explanation = self.gemini.get_scientific_explanation(response)
                            response_text += f"🔬 Scientific Analysis of New Results:\n{explanation}"
                else:
                    response_text = "📊 Analysis has converged - no further iterations needed."
                
                st.session_state.chat_history.append({
                    "type": "agent",
                    "content": response_text,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                return
            else:
                st.warning("No previous analysis to continue from.")
                return
            
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Use Gemini for enhanced understanding if available
        if self.gemini:
            with st.spinner("🧠 Analyzing with Gemini AI..."):
                gemini_response = await self.gemini.process_query(user_input)
                action = gemini_response["action"]
                parameters = gemini_response["parameters"]
                explanation = gemini_response["explanation"]
                
                # Store for potential continuation
                st.session_state.last_action = action
                st.session_state.last_parameters = parameters
                
                # Add Gemini's explanation to chat
                st.session_state.chat_history.append({
                    "type": "agent",
                    "content": f"💡 {explanation}",
                    "timestamp": datetime.datetime.now().isoformat()
                })
        else:
            # Fallback to rule-based parameter extraction
            params = self._extract_parameters(user_input)
            action = params["action"]
            parameters = params["parameters"]
            st.session_state.last_action = action
            st.session_state.last_parameters = parameters
        
        # Process with agent
        with st.spinner("🔬 Processing analysis..."):
            response = self.agent.execute_action(
                action_name=action,
                parameters=parameters
            )
            st.session_state.last_results = response
        
        # Format response and get scientific explanation
        response_text = self._format_response(response)
        if self.gemini and "error" not in response:
            explanation = self.gemini.get_scientific_explanation(response)
            response_text += f"\n\n🔬 Scientific Analysis:\n{explanation}"
            
            # Add iteration suggestion if appropriate
            should_continue = self.gemini.should_continue_iteration(response)
            if should_continue:
                response_text += "\n\n💡 The analysis could benefit from another iteration. Type 'continue' to proceed."
        
        # Add agent response to chat history
        st.session_state.chat_history.append({
            "type": "agent",
            "content": response_text,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Display results with visualizations
        if "error" not in response:
            self._display_results(response, action)
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract action and parameters from natural language input (fallback method)"""
        text = text.lower()
        params = {"parameters": {}}
        
        # Extract Morgan fingerprint parameters if specified
        radius = 2  # default
        n_bits = 2048  # default
        
        if "radius" in text:
            for word in text.split():
                if word.isdigit() and int(word) in range(1, 5):
                    radius = int(word)
                    break
        
        if "bits" in text:
            for word in text.split():
                if word.isdigit() and int(word) in [512, 1024, 2048, 4096]:
                    n_bits = int(word)
                    break
        
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
                if ("(" in word and ")" in word) or "=" in word:
                    params["parameters"]["smiles"] = word
                    break
            
            # Add fingerprint parameters
            params["parameters"]["radius"] = radius
            params["parameters"]["n_bits"] = n_bits
            
            # Extract threshold if specified
            if "threshold" in text:
                for word in text.split():
                    try:
                        value = float(word)
                        if 0 <= value <= 1:
                            params["parameters"]["threshold"] = value
                            break
                    except ValueError:
                        continue
            else:
                params["parameters"]["threshold"] = 0.7  # Default threshold
            
        else:
            # Default to complete pipeline
            params["action"] = "complete_pipeline"
            params["parameters"] = {
                "uniprot_ids": [],
                "smiles": "",
                "radius": radius,
                "n_bits": n_bits,
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