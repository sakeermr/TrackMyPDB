import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import datetime
from .heteroatom_extractor import HeteroatomExtractor
import streamlit as st

# Import RDKit first to check availability
try:
    import rdkit
    from .similarity_analyzer import SimilarityAnalyzer
except ImportError:
    from .similarity_analyzer_simple import SimilarityAnalyzer
    st.warning("⚠️ RDKit not available - using simplified molecular similarity")

class AgentAction(Enum):
    EXTRACT_HETEROATOMS = "extract_heteroatoms"
    ANALYZE_SIMILARITY = "analyze_similarity"
    COMPLETE_PIPELINE = "complete_pipeline"
    INTERPRET_RESULTS = "interpret_results"
    SUGGEST_FOLLOWUP = "suggest_followup"

@dataclass
class AgentQuery:
    text: str
    timestamp: datetime.datetime
    query_type: str

@dataclass
class AgentResponse:
    action: AgentAction
    parameters: Dict[str, Any]
    explanation: str
    confidence: float
    follow_up_suggestions: List[str]

class TrackMyPDBAgent:
    def __init__(self):
        self.heteroatom_extractor = HeteroatomExtractor()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.query_history = []
    
    def add_to_history(self, query: AgentQuery):
        """Add a query to the history"""
        self.query_history.append(query)

    @staticmethod
    def validate_query(query: AgentQuery) -> bool:
        """Validate a query"""
        if not query.text:
            raise ValueError("Query text cannot be empty")
        if query.query_type not in ["heteroatom_analysis", "similarity_analysis", "complete_pipeline"]:
            raise ValueError(f"Invalid query type: {query.query_type}")
        return True

    def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action with given parameters"""
        if action_name == "extract_heteroatoms":
            uniprot_ids = parameters.get("uniprot_ids", [])
            results = self.heteroatom_extractor.extract_heteroatoms(uniprot_ids)
            return {"results": results}
        
        elif action_name == "analyze_similarity":
            smiles = parameters.get("smiles", "")
            threshold = parameters.get("threshold", 0.7)
            radius = parameters.get("radius", 2)
            n_bits = parameters.get("n_bits", 2048)
            
            if hasattr(self, "last_heteroatom_results"):
                # Create new analyzer with custom parameters
                analyzer = SimilarityAnalyzer(radius=radius, n_bits=n_bits)
                results = analyzer.analyze_similarity(
                    target_smiles=smiles,
                    heteroatom_df=self.last_heteroatom_results,
                    min_similarity=threshold
                )
                return {"results": results}
            else:
                return {"error": "Please run heteroatom extraction first"}
        
        elif action_name == "complete_pipeline":
            uniprot_ids = parameters.get("uniprot_ids", [])
            smiles = parameters.get("smiles", "")
            threshold = parameters.get("threshold", 0.7)
            radius = parameters.get("radius", 2)
            n_bits = parameters.get("n_bits", 2048)
            
            # First extract heteroatoms
            heteroatom_results = self.heteroatom_extractor.extract_heteroatoms(uniprot_ids)
            self.last_heteroatom_results = heteroatom_results
            
            # Create new analyzer with custom parameters
            analyzer = SimilarityAnalyzer(radius=radius, n_bits=n_bits)
            
            # Then analyze similarity
            similarity_results = analyzer.analyze_similarity(
                target_smiles=smiles,
                heteroatom_df=heteroatom_results,
                min_similarity=threshold
            )
            
            return {
                "heteroatom_results": heteroatom_results,
                "similarity_results": similarity_results
            }
        
        else:
            return {"error": f"Unknown action: {action_name}"}

    async def process_query(self, query: AgentQuery) -> Dict[str, Any]:
        """Process a query and return results"""
        self.validate_query(query)
        self.add_to_history(query)
        
        # Determine action based on query type
        if query.query_type == "heteroatom_analysis":
            return await self._process_heteroatom_query(query)
        elif query.query_type == "similarity_analysis":
            return await self._process_similarity_query(query)
        else:
            return await self._process_pipeline_query(query)

    async def _process_heteroatom_query(self, query: AgentQuery) -> Dict[str, Any]:
        """Process a heteroatom analysis query"""
        return {
            "query_type": "heteroatom_analysis",
            "status": "success",
            "timestamp": query.timestamp.isoformat()
        }

    async def _process_similarity_query(self, query: AgentQuery) -> Dict[str, Any]:
        """Process a similarity analysis query"""
        return {
            "query_type": "similarity_analysis",
            "status": "success",
            "timestamp": query.timestamp.isoformat()
        }

    async def _process_pipeline_query(self, query: AgentQuery) -> Dict[str, Any]:
        """Process a complete pipeline query"""
        return {
            "query_type": "complete_pipeline",
            "status": "success",
            "timestamp": query.timestamp.isoformat()
        }