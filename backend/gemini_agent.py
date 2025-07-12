"""
TrackMyPDB Gemini AI Agent
@author Anu Gamage

This module provides Gemini AI integration for enhanced natural language understanding
and scientific analysis in TrackMyPDB.
Licensed under MIT License - Open Source Project
"""

import os
import re
import json
from typing import Dict, Any, Optional
import streamlit as st
import asyncio
import time

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.warning("Google Generative AI not installed. Install with: pip install google-generativeai")

from .config import Config

class GeminiAgent:
    """
    A class that handles Gemini AI integration for enhanced natural language processing
    and scientific analysis in TrackMyPDB.
    """
    
    def __init__(self):
        """Initialize the Gemini AI agent"""
        self.model = None
        self.api_key = None
        self.is_initialized = False
        
        if not GENAI_AVAILABLE:
            st.error("❌ Google Generative AI library not available")
            return
        
        # Get API key from config
        self.api_key = Config.get_api_key()
        
        if not self.api_key:
            st.error("❌ No Google AI Studio API key found")
            return
        
        try:
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=Config.GEMINI_TEMPERATURE,
                    max_output_tokens=Config.GEMINI_MAX_OUTPUT_TOKENS,
                )
            )
            
            self.is_initialized = True
            st.success("✅ Gemini AI initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini AI: {str(e)}")
            # Try fallback model
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_initialized = True
                st.warning("⚠️ Using fallback Gemini model")
            except Exception as e2:
                st.error(f"❌ Fallback model also failed: {str(e2)}")

    def is_available(self) -> bool:
        """Check if Gemini AI is available and properly initialized"""
        return GENAI_AVAILABLE and self.is_initialized and self.model is not None

    async def generate_ai_response(self, prompt: str) -> str:
        """Generate AI response with proper error handling"""
        if not self.is_available():
            return "AI not available. Using fallback analysis."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text if response.text else "No response generated"
        except Exception as e:
            st.warning(f"AI generation error: {str(e)}")
            return f"AI error: {str(e)}"

    def generate_ai_response_sync(self, prompt: str) -> str:
        """Synchronous wrapper for AI response generation"""
        if not self.is_available():
            return "AI not available. Using fallback analysis."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text if response.text else "No response generated"
        except Exception as e:
            return f"AI error: {str(e)}"

    def process_query_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous processing of natural language queries"""
        if not self.is_available():
            return self._fallback_parse(query)
        
        try:
            prompt = self._create_analysis_prompt(query)
            response = self.generate_ai_response_sync(prompt)
            return self._parse_gemini_response(response, query)
        except Exception as e:
            st.warning(f"Query processing error: {str(e)}")
            return self._fallback_parse(query)

    def _create_analysis_prompt(self, query: str) -> str:
        """Create analysis prompt for Gemini"""
        return f"""
        You are a molecular biology AI assistant specializing in protein-drug interactions and structural analysis.
        
        Analyze this user query and determine the appropriate action:
        Query: "{query}"
        
        Available actions:
        1. extract_heteroatoms - Extract heteroatoms from protein structures
        2. similarity_analysis - Perform molecular similarity analysis
        3. combined_analysis - Both heteroatom extraction and similarity analysis
        
        Extract any relevant parameters like:
        - UniProt IDs (format: P12345)
        - SMILES structures 
        - Similarity thresholds (0.0-1.0)
        - PDB IDs
        
        Respond in JSON format:
        {{
            "action": "action_name",
            "parameters": {{"key": "value"}},
            "explanation": "brief explanation",
            "confidence": 0.9
        }}
        """

    def _parse_gemini_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                # If no JSON found, parse manually
                return self._manual_parse_response(response, original_query)
        except Exception as e:
            return self._fallback_parse(original_query)

    def _manual_parse_response(self, response: str, query: str) -> Dict[str, Any]:
        """Manually parse AI response when JSON parsing fails"""
        response_lower = response.lower()
        
        if "heteroatom" in response_lower:
            action = "extract_heteroatoms"
        elif "similarity" in response_lower:
            action = "similarity_analysis"
        else:
            action = "combined_analysis"
        
        # Extract parameters from original query
        uniprot_match = re.findall(r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}\b', query.upper())
        smiles_match = re.search(r'[A-Za-z0-9@+\-\[\]()=:#$.\/\\]{3,}', query)
        
        parameters = {}
        if uniprot_match:
            parameters["uniprot_ids"] = uniprot_match
        if smiles_match:
            parameters["target_smiles"] = smiles_match.group()
        
        return {
            "action": action,
            "parameters": parameters,
            "explanation": f"Parsed from AI response: {action}",
            "confidence": 0.8
        }
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback parsing when AI is not available"""
        query_lower = query.lower()
        
        # Extract UniProt IDs
        uniprot_match = re.findall(r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}\b', query.upper())
        
        # Extract SMILES-like patterns
        smiles_patterns = [
            r'SMILES[:\s]+([A-Za-z0-9@+\-\[\]()=:#$.\/\\]+)',
            r'\b([A-Za-z0-9@+\-\[\]()=:#$.\/\\]{5,})\b'
        ]
        
        smiles = None
        for pattern in smiles_patterns:
            match = re.search(pattern, query)
            if match:
                candidate = match.group(1) if len(match.groups()) > 0 else match.group(0)
                if any(char in candidate for char in ['C', 'N', 'O', '=', '(', ')', '[', ']']):
                    smiles = candidate
                    break
        
        # Determine action based on content
        if "heteroatom" in query_lower and smiles:
            action = "combined_analysis"
        elif "heteroatom" in query_lower or "extract" in query_lower:
            action = "extract_heteroatoms"
        elif "similarity" in query_lower or "compare" in query_lower:
            action = "similarity_analysis"
        else:
            action = "extract_heteroatoms"  # Default
        
        parameters = {}
        if uniprot_match:
            parameters["uniprot_ids"] = uniprot_match
        if smiles:
            parameters["target_smiles"] = smiles
        
        return {
            "action": action,
            "parameters": parameters,
            "explanation": f"Fallback parsing detected {action}",
            "confidence": 0.6
        }

    def get_scientific_explanation(self, results: Dict[str, Any]) -> str:
        """Get scientific explanation of analysis results"""
        if not self.is_available():
            return "AI explanation not available. Results processed successfully."
        
        try:
            # Prepare results summary
            result_summary = self._prepare_results_summary(results)
            
            prompt = f"""
            As a molecular biology expert, provide a concise scientific explanation of these protein-ligand analysis results:
            
            {result_summary}
            
            Focus on:
            1. Key molecular findings and patterns
            2. Potential biological significance
            3. Structural insights
            4. Recommendations for further study
            
            Keep the explanation scientific but accessible, around 3-4 sentences.
            """
            
            return self.generate_ai_response_sync(prompt)
            
        except Exception as e:
            return f"Scientific explanation generation failed: {str(e)}"

    def _prepare_results_summary(self, results: Dict[str, Any]) -> str:
        """Prepare a concise summary of results for AI analysis"""
        summary_parts = []
        
        if isinstance(results, dict):
            for key, value in results.items():
                if key in ['total_heteroatoms', 'unique_pdbs', 'similarity_matches']:
                    summary_parts.append(f"{key}: {value}")
                elif key == 'top_similarities' and isinstance(value, list):
                    summary_parts.append(f"Top similarity scores: {value[:3]}")
        
        return "; ".join(summary_parts) if summary_parts else str(results)[:500]

    def should_continue_iteration(self, current_results: Dict[str, Any]) -> bool:
        """Determine if analysis iteration should continue"""
        if not self.is_available():
            return False  # Conservative fallback
        
        try:
            result_summary = self._prepare_results_summary(current_results)
            
            prompt = f"""
            Analyze these molecular analysis results and determine if further iteration would provide significant additional insights:
            
            {result_summary}
            
            Consider:
            1. Convergence and completeness of current results
            2. Quality and diversity of findings
            3. Potential for discovering new patterns
            
            Respond with only 'true' or 'false'.
            """
            
            response = self.generate_ai_response_sync(prompt)
            return response.strip().lower() == 'true'
            
        except Exception as e:
            return False  # Conservative fallback

    def validate_setup(self) -> tuple[bool, str]:
        """Validate the Gemini setup"""
        return Config.validate_ai_setup()