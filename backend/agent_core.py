import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import datetime
from .heteroatom_extractor import OptimizedHeteroatomExtractor
import streamlit as st

# Import the new agentic layer
from .agentic_layer import (
    TrackMyPDBAgenticInterface, 
    AgentMode, 
    AnalysisType
)

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
    AGENTIC_ANALYSIS = "agentic_analysis"  # New agentic analysis

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
        self.heteroatom_extractor = OptimizedHeteroatomExtractor()
        # Initialize with default settings
        self.similarity_analyzer = SimilarityAnalyzer()
        self.query_history = []
        
        # Initialize the new agentic interface
        self.agentic_interface = TrackMyPDBAgenticInterface()

    def add_to_history(self, query: AgentQuery):
        """Add a query to the history"""
        self.query_history.append(query)

    @staticmethod
    def validate_query(query: AgentQuery) -> bool:
        """Validate a query"""
        if not query.text:
            raise ValueError("Query text cannot be empty")
        if query.query_type not in ["heteroatom_analysis", "similarity_analysis", "complete_pipeline", "agentic_analysis"]:
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
            fp_type = parameters.get("fp_type", "morgan")
            metric = parameters.get("metric", "tanimoto")
            
            if hasattr(self, "last_heteroatom_results"):
                # Create new analyzer with custom parameters
                analyzer = SimilarityAnalyzer(
                    radius=radius,
                    n_bits=n_bits,
                    fp_type=fp_type,
                    metric=metric
                )
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
            fp_type = parameters.get("fp_type", "morgan")
            metric = parameters.get("metric", "tanimoto")
            
            # First extract heteroatoms
            heteroatom_results = self.heteroatom_extractor.extract_heteroatoms(uniprot_ids)
            self.last_heteroatom_results = heteroatom_results
            
            # Create analyzer with custom parameters
            analyzer = SimilarityAnalyzer(
                radius=radius,
                n_bits=n_bits,
                fp_type=fp_type,
                metric=metric
            )
            
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
        
        elif action_name == "agentic_analysis":
            # New agentic analysis action
            return self._execute_agentic_analysis(parameters)
        
        else:
            return {"error": f"Unknown action: {action_name}"}
    
    def _execute_agentic_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agentic analysis using the new agentic layer"""
        try:
            # Extract parameters
            target_smiles = parameters.get("target_smiles", "")
            mode_str = parameters.get("mode", "ai_assisted")
            analysis_types_str = parameters.get("analysis_types", ["morgan_similarity", "drug_likeness"])
            
            # Convert mode string to enum
            mode = AgentMode.AI_ASSISTED
            if mode_str == "manual":
                mode = AgentMode.MANUAL
            elif mode_str == "fully_autonomous":
                mode = AgentMode.FULLY_AUTONOMOUS
            
            # Convert analysis types
            analysis_types = []
            for analysis_type_str in analysis_types_str:
                if analysis_type_str == "morgan_similarity":
                    analysis_types.append(AnalysisType.MORGAN_SIMILARITY)
                elif analysis_type_str == "tanimoto_similarity":
                    analysis_types.append(AnalysisType.TANIMOTO_SIMILARITY)
                elif analysis_type_str == "drug_likeness":
                    analysis_types.append(AnalysisType.DRUG_LIKENESS)
            
            # Get heteroatom data if available
            heteroatom_data = getattr(self, 'last_heteroatom_results', None)
            
            # Run async analysis in sync context (for Streamlit compatibility)
            import asyncio
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new thread for the async operation
                    import threading
                    import concurrent.futures
                    
                    def run_async():
                        # Create new event loop in thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self.agentic_interface.run_comprehensive_analysis(
                                    target_smiles=target_smiles,
                                    mode=mode,
                                    analysis_types=analysis_types,
                                    heteroatom_data=heteroatom_data,
                                    **parameters
                                )
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async)
                        results = future.result(timeout=60)  # 60 second timeout
                else:
                    # No running loop, can use directly
                    results = loop.run_until_complete(
                        self.agentic_interface.run_comprehensive_analysis(
                            target_smiles=target_smiles,
                            mode=mode,
                            analysis_types=analysis_types,
                            heteroatom_data=heteroatom_data,
                            **parameters
                        )
                    )
            except RuntimeError:
                # No event loop exists, create one
                results = asyncio.run(
                    self.agentic_interface.run_comprehensive_analysis(
                        target_smiles=target_smiles,
                        mode=mode,
                        analysis_types=analysis_types,
                        heteroatom_data=heteroatom_data,
                        **parameters
                    )
                )
            
            return {"agentic_results": results}
            
        except Exception as e:
            return {"error": f"Agentic analysis failed: {str(e)}"}

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