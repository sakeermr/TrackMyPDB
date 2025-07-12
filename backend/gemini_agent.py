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
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import asyncio
from .config import Config

class GeminiAgent:
    """
    A class that handles Gemini AI integration for enhanced natural language processing
    and scientific analysis in TrackMyPDB.
    """
    
    def __init__(self):
        """Initialize the Gemini AI agent"""
        # Try multiple environment variable names for flexibility
        api_key = (
            os.getenv('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_API_KEY') or 
            getattr(Config, 'GEMINI_API_KEY', None) or
            getattr(Config, 'GOOGLE_API_KEY', None)
        )
        
        if not api_key:
            raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        
        # Get the Gemini model
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {e}")

    def process_query_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for process_query"""
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use run_in_executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.process_query(query))
                        return future.result()
                else:
                    return loop.run_until_complete(self.process_query(query))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(self.process_query(query))
        except Exception as e:
            # Fallback to simple parsing if Gemini fails
            return self._fallback_parse(query)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query using Gemini"""
        try:
            prompt = self._create_analysis_prompt(query)
            response = await self._generate_response(prompt)
            return self._parse_gemini_response(response, query)
        except Exception as e:
            # Fallback to simple parsing
            return self._fallback_parse(query)
    
    def _fallback_parse(self, query: str) -> Dict[str, Any]:
        """Fallback parsing when Gemini is not available"""
        query_lower = query.lower()
        
        if "heteroatom" in query_lower or "extract" in query_lower:
            # Extract protein names using simple regex
            protein_match = re.search(r'\b([A-Z0-9]{3,6})\b', query.upper())
            protein_name = protein_match.group(1) if protein_match else "1ABC"
            
            return {
                "action": "extract_heteroatoms",
                "parameters": {"protein_name": protein_name},
                "explanation": f"Extracting heteroatoms from protein {protein_name}"
            }
        elif "similarity" in query_lower or "compare" in query_lower:
            return {
                "action": "similarity_analysis", 
                "parameters": {"threshold": 0.8},
                "explanation": "Performing similarity analysis with default threshold"
            }
        else:
            return {
                "action": "extract_heteroatoms",
                "parameters": {"protein_name": "1ABC"},
                "explanation": "Default action: extracting heteroatoms"
            }

    def get_scientific_explanation(self, results: Dict[str, Any]) -> str:
        """
        Get a scientific explanation of analysis results
        
        Args:
            results (dict): Analysis results
            
        Returns:
            str: Scientific explanation of the results
        """
        try:
            # Format results for the prompt
            result_summary = str(results)[:1000]  # Truncate to avoid token limits
            
            prompt = f"""
            Analyze these protein-ligand analysis results and provide a brief scientific explanation:
            {result_summary}
            
            Focus on:
            1. Key findings and patterns
            2. Potential biological significance
            3. Suggestions for further investigation
            
            Keep the explanation concise and scientific.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Unable to generate scientific explanation: {e}"

    def should_continue_iteration(self, current_results: Dict[str, Any]) -> bool:
        """
        Determine if analysis iteration should continue
        
        Args:
            current_results (dict): Current analysis results
            
        Returns:
            bool: Whether to continue iteration
        """
        try:
            # Analyze current results
            result_summary = str(current_results)[:1000]
            
            prompt = f"""
            Analyze these protein analysis results and determine if further iteration would be beneficial:
            {result_summary}
            
            Consider:
            1. Convergence of results
            2. Quality of findings
            3. Potential for new insights
            
            Respond with just 'true' or 'false'.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip().lower() == 'true'
            
        except Exception as e:
            st.warning(f"Error in iteration analysis: {e}")
            return False