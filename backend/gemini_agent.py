"""
TrackMyPDB Gemini AI Integration
Enhances natural language understanding using Google's Gemini model
"""

import google.generativeai as genai
import streamlit as st
from typing import Dict, Any, List, Optional
import json
import os

# Default API key storage location
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.py')

class GeminiAgent:
    def __init__(self, api_key: str = None):
        self._initialize_api_key(api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Define system prompt for protein analysis
        self.system_prompt = """You are an expert protein analysis assistant specializing in:
1. Heteroatom extraction from protein structures
2. Molecular similarity analysis
3. Structure-activity relationships
4. Protein-ligand interactions

Format your responses as JSON with the following structure:
{
    "action": "extract_heteroatoms" | "analyze_similarity" | "complete_pipeline",
    "parameters": {
        "uniprot_ids": ["P53", ...],  // For heteroatom extraction
        "smiles": "CC(=O)O",          // For similarity analysis
        "radius": 2,                  // Morgan fingerprint radius
        "n_bits": 2048,              // Number of bits
        "threshold": 0.7             // Similarity threshold
    },
    "explanation": "Brief explanation of what you're doing",
    "follow_up_suggestions": ["Suggestion 1", "Suggestion 2"]
}
"""
        self.chat = self.model.start_chat(history=[])
        self._initialize_chat()

    def _initialize_api_key(self, api_key: str = None):
        """Initialize Gemini API key from provided key or config file"""
        if api_key:
            genai.configure(api_key=api_key)
            self._save_api_key(api_key)  # Save for future use
        else:
            # Try to load from config file
            try:
                from . import config
                if hasattr(config, 'GEMINI_API_KEY'):
                    genai.configure(api_key=config.GEMINI_API_KEY)
                else:
                    raise ValueError("No API key found in config.py")
            except ImportError:
                self._create_config_file()
                raise ValueError("Please provide a Gemini API key")

    def _save_api_key(self, api_key: str):
        """Save API key to config file"""
        config_content = f"GEMINI_API_KEY = '{api_key}'\n"
        try:
            with open(CONFIG_FILE, 'w') as f:
                f.write(config_content)
        except Exception as e:
            st.warning(f"Could not save API key to config file: {e}")

    def _create_config_file(self):
        """Create config file template if it doesn't exist"""
        if not os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'w') as f:
                    f.write("# Gemini API Configuration\nGEMINI_API_KEY = ''\n")
            except Exception as e:
                st.warning(f"Could not create config file: {e}")

    def _initialize_chat(self):
        """Initialize chat with system prompt"""
        self.chat.send_message(self.system_prompt)

    def _validate_json_response(self, text: str) -> Dict[str, Any]:
        """Validate and extract JSON from response"""
        try:
            # Find JSON block in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "action": "complete_pipeline",
            "parameters": {},
            "explanation": "Could not parse response",
            "follow_up_suggestions": []
        }

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query through Gemini"""
        try:
            # Add protein analysis context
            enhanced_query = f"""Analyze this protein analysis request: {query}

Consider:
- UniProt IDs start with P, Q, or O and contain numbers
- SMILES strings contain parentheses or equals signs
- Morgan fingerprint radius should be 1-4
- Number of bits should be 512, 1024, 2048, or 4096
- Similarity threshold should be 0.0-1.0

Respond with JSON only."""

            response = self.chat.send_message(enhanced_query)
            
            # Parse and validate response
            result = self._validate_json_response(response.text)
            
            # Add default parameters if missing
            if "parameters" not in result:
                result["parameters"] = {}
            if "radius" not in result["parameters"]:
                result["parameters"]["radius"] = 2
            if "n_bits" not in result["parameters"]:
                result["parameters"]["n_bits"] = 2048
            if "threshold" not in result["parameters"]:
                result["parameters"]["threshold"] = 0.7
                
            return result

        except Exception as e:
            st.error(f"Error processing query with Gemini: {str(e)}")
            return {
                "action": "complete_pipeline",
                "parameters": {},
                "explanation": f"Error: {str(e)}",
                "follow_up_suggestions": []
            }

    def get_scientific_explanation(self, results: Dict[str, Any]) -> str:
        """Generate scientific explanation of results"""
        try:
            # Format results summary for Gemini
            results_summary = json.dumps(results, indent=2)
            
            prompt = f"""Explain these protein analysis results scientifically:
{results_summary}

Focus on:
1. Structural insights
2. Chemical properties
3. Biological implications
4. Key findings and patterns

Keep it concise and technical."""

            response = self.chat.send_message(prompt)
            return response.text

        except Exception as e:
            return f"Could not generate explanation: {str(e)}"