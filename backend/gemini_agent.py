"""
TrackMyPDB Gemini AI Agent
@author Anu Gamage

This module provides Gemini AI integration for enhanced natural language understanding
and scientific analysis in TrackMyPDB.
Licensed under MIT License - Open Source Project
"""

import os
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

class GeminiAgent:
    """
    A class that handles Gemini AI integration for enhanced natural language processing
    and scientific analysis in TrackMyPDB.
    """
    
    def __init__(self):
        """Initialize the Gemini AI agent"""
        load_dotenv()  # Load environment variables
        
        # Configure Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        
        # Get the Gemini model
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {e}")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and determine appropriate actions
        
        Args:
            query (str): The user's natural language query
            
        Returns:
            dict: Action parameters extracted from the query
        """
        try:
            # Prompt engineering for better extraction
            prompt = f"""
            Analyze this protein analysis query: "{query}"
            
            Extract:
            1. Main action (extract_heteroatoms/analyze_similarity/complete_pipeline)
            2. Parameters (UniProt IDs, SMILES, threshold, etc.)
            3. A brief explanation of what will be done
            
            Format response as JSON with keys:
            - action: string
            - parameters: object
            - explanation: string
            """
            
            response = await self.model.generate_content(prompt)
            result = response.text
            
            # Parse the JSON response
            try:
                parsed = eval(result)  # Safe since we control the input
                return parsed
            except:
                # Fallback to basic extraction
                return {
                    "action": "extract_heteroatoms",
                    "parameters": {},
                    "explanation": "Analyzing protein structure..."
                }
                
        except Exception as e:
            st.error(f"Error processing query with Gemini: {e}")
            return {
                "action": "extract_heteroatoms",
                "parameters": {},
                "explanation": "Analyzing protein structure..."
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