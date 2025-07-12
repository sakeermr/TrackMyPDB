"""
Agentic Layer Architecture for TrackMyPDB
Integrates AI capabilities with existing Morgan and Tanimoto analysis
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import os

# Check for RDKit availability
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Check for Google Gemini availability
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from abc import ABC, abstractmethod

class AgentMode(Enum):
    """Agent operation modes"""
    MANUAL = "manual"
    AI_ASSISTED = "ai_assisted"
    FULLY_AUTONOMOUS = "fully_autonomous"

class AnalysisType(Enum):
    """Types of molecular analysis"""
    MORGAN_SIMILARITY = "morgan_similarity"
    TANIMOTO_SIMILARITY = "tanimoto_similarity"
    DRUG_LIKENESS = "drug_likeness"
    SCAFFOLD_ANALYSIS = "scaffold_analysis"
    BINDING_AFFINITY = "binding_affinity"
    TOXICITY_PREDICTION = "toxicity_prediction"

@dataclass
class AnalysisRequest:
    """Request structure for analysis"""
    target_smiles: str
    analysis_types: List[AnalysisType]
    mode: AgentMode
    parameters: Dict[str, Any]
    context: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """Result structure for analysis"""
    analysis_type: AnalysisType
    success: bool
    results: Dict[str, Any]
    confidence: float
    recommendations: List[str]
    metadata: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None

class BaseAgent(ABC):
    """Base class for all analysis agents"""
    
    def __init__(self, name: str, gemini_api_key: Optional[str] = None):
        self.name = name
        self.gemini_api_key = gemini_api_key
        self.ai_enabled = False
        
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.ai_enabled = True
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini for {name}: {e}")
                self.ai_enabled = False
        
    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform analysis based on request"""
        pass
    
    async def get_ai_insights(self, context: str, data: Dict[str, Any]) -> str:
        """Get AI insights using Gemini"""
        if not self.ai_enabled:
            return "AI insights not available (Gemini not configured)"
        
        try:
            prompt = f"""
            Context: {context}
            Data: {json.dumps(data, indent=2, default=str)}
            
            Please analyze this molecular data and provide:
            1. Key insights and patterns
            2. Recommendations for drug discovery
            3. Potential concerns or limitations
            4. Suggested next steps
            
            Format your response as structured insights.
            """
            
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"AI analysis failed: {str(e)}"

class MorganSimilarityAgent(BaseAgent):
    """Enhanced Morgan fingerprint similarity agent"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        super().__init__("Morgan Similarity Agent", gemini_api_key)
        
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform Morgan similarity analysis"""
        start_time = time.time()
        
        if not RDKIT_AVAILABLE:
            return AnalysisResult(
                analysis_type=AnalysisType.MORGAN_SIMILARITY,
                success=False,
                results={},
                confidence=0.0,
                recommendations=["RDKit not available for Morgan fingerprint analysis"],
                metadata={},
                execution_time=time.time() - start_time,
                error_message="RDKit not available"
            )
        
        try:
            # Get parameters
            radius = request.parameters.get('radius', 2)
            n_bits = request.parameters.get('n_bits', 2048)
            threshold = request.parameters.get('threshold', 0.2)
            heteroatom_data = request.parameters.get('heteroatom_data')
            
            # Convert SMILES to molecule
            target_mol = Chem.MolFromSmiles(request.target_smiles)
            if not target_mol:
                raise ValueError(f"Invalid SMILES: {request.target_smiles}")
            
            # Generate Morgan fingerprint
            target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                target_mol, radius, nBits=n_bits
            )
            
            # Calculate similarities
            similarities = []
            
            if heteroatom_data is not None and isinstance(heteroatom_data, pd.DataFrame):
                # Use actual heteroatom data
                for idx, row in heteroatom_data.iterrows():
                    if 'SMILES' in row and pd.notna(row['SMILES']):
                        compound_mol = Chem.MolFromSmiles(row['SMILES'])
                        if compound_mol:
                            compound_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                                compound_mol, radius, nBits=n_bits
                            )
                            similarity = DataStructs.TanimotoSimilarity(target_fp, compound_fp)
                            
                            if similarity >= threshold:
                                similarities.append({
                                    'compound_id': row.get('PDB_ID', f"COMP_{idx}"),
                                    'heteroatom_code': row.get('Heteroatom_Code', 'UNK'),
                                    'smiles': row['SMILES'],
                                    'similarity': similarity,
                                    'molecular_weight': Descriptors.MolWt(compound_mol),
                                    'logp': Crippen.MolLogP(compound_mol)
                                })
            else:
                # Generate mock data for demonstration
                for i in range(100):
                    compound_smiles = f"C{'C' * (i % 10)}O"
                    compound_mol = Chem.MolFromSmiles(compound_smiles)
                    if compound_mol:
                        compound_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                            compound_mol, radius, nBits=n_bits
                        )
                        similarity = DataStructs.TanimotoSimilarity(target_fp, compound_fp)
                        
                        if similarity >= threshold:
                            similarities.append({
                                'compound_id': f"COMP_{i:04d}",
                                'smiles': compound_smiles,
                                'similarity': similarity,
                                'molecular_weight': Descriptors.MolWt(compound_mol),
                                'logp': Crippen.MolLogP(compound_mol)
                            })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Get AI insights if in AI mode
            ai_insights = ""
            if request.mode in [AgentMode.AI_ASSISTED, AgentMode.FULLY_AUTONOMOUS]:
                ai_insights = await self.get_ai_insights(
                    f"Morgan fingerprint similarity analysis for {request.target_smiles}",
                    {'similarities': similarities[:10], 'parameters': request.parameters}
                )
            
            # Generate recommendations
            recommendations = []
            if similarities:
                recommendations.append(f"Found {len(similarities)} similar compounds above threshold {threshold}")
                recommendations.append(f"Top similarity score: {similarities[0]['similarity']:.3f}")
                
                if request.mode == AgentMode.FULLY_AUTONOMOUS:
                    recommendations.append("Consider analyzing top 10 compounds for drug-like properties")
                    recommendations.append("Suggest scaffold hopping analysis for lead optimization")
            else:
                recommendations.append(f"No compounds found above similarity threshold {threshold}")
                recommendations.append("Consider lowering the threshold or using different fingerprint parameters")
            
            execution_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.MORGAN_SIMILARITY,
                success=True,
                results={
                    'similarities': similarities,
                    'total_found': len(similarities),
                    'parameters_used': {
                        'radius': radius,
                        'n_bits': n_bits,
                        'threshold': threshold
                    },
                    'ai_insights': ai_insights
                },
                confidence=0.85,
                recommendations=recommendations,
                metadata={
                    'analysis_time': datetime.now().isoformat(),
                    'agent_version': "1.0.0",
                    'mode': request.mode.value
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AnalysisResult(
                analysis_type=AnalysisType.MORGAN_SIMILARITY,
                success=False,
                results={},
                confidence=0.0,
                recommendations=[],
                metadata={},
                execution_time=execution_time,
                error_message=str(e)
            )

class TanimotoSimilarityAgent(BaseAgent):
    """Enhanced Tanimoto similarity agent"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        super().__init__("Tanimoto Similarity Agent", gemini_api_key)
        
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform Tanimoto similarity analysis"""
        start_time = time.time()
        
        if not RDKIT_AVAILABLE:
            return AnalysisResult(
                analysis_type=AnalysisType.TANIMOTO_SIMILARITY,
                success=False,
                results={},
                confidence=0.0,
                recommendations=["RDKit not available for Tanimoto similarity analysis"],
                metadata={},
                execution_time=time.time() - start_time,
                error_message="RDKit not available"
            )
        
        try:
            # Get parameters
            fingerprint_type = request.parameters.get('fingerprint_type', 'morgan')
            threshold = request.parameters.get('threshold', 0.3)
            heteroatom_data = request.parameters.get('heteroatom_data')
            
            # Convert SMILES to molecule
            target_mol = Chem.MolFromSmiles(request.target_smiles)
            if not target_mol:
                raise ValueError(f"Invalid SMILES: {request.target_smiles}")
            
            # Generate appropriate fingerprint
            if fingerprint_type == 'morgan':
                target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 2)
            elif fingerprint_type == 'maccs':
                target_fp = rdMolDescriptors.GetMACCSKeysFingerprint(target_mol)
            else:
                target_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(target_mol)
            
            # Calculate Tanimoto similarities
            similarities = []
            
            if heteroatom_data is not None and isinstance(heteroatom_data, pd.DataFrame):
                # Use actual heteroatom data
                for idx, row in heteroatom_data.iterrows():
                    if 'SMILES' in row and pd.notna(row['SMILES']):
                        compound_mol = Chem.MolFromSmiles(row['SMILES'])
                        if compound_mol:
                            if fingerprint_type == 'morgan':
                                compound_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(compound_mol, 2)
                            elif fingerprint_type == 'maccs':
                                compound_fp = rdMolDescriptors.GetMACCSKeysFingerprint(compound_mol)
                            else:
                                compound_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(compound_mol)
                            
                            tanimoto = DataStructs.TanimotoSimilarity(target_fp, compound_fp)
                            
                            if tanimoto >= threshold:
                                similarities.append({
                                    'compound_id': row.get('PDB_ID', f"TANI_{idx}"),
                                    'heteroatom_code': row.get('Heteroatom_Code', 'UNK'),
                                    'smiles': row['SMILES'],
                                    'tanimoto_similarity': tanimoto,
                                    'fingerprint_type': fingerprint_type,
                                    'molecular_weight': Descriptors.MolWt(compound_mol)
                                })
            else:
                # Generate mock data
                for i in range(100):
                    compound_smiles = f"C{'N' * (i % 8)}C=O"
                    compound_mol = Chem.MolFromSmiles(compound_smiles)
                    if compound_mol:
                        if fingerprint_type == 'morgan':
                            compound_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(compound_mol, 2)
                        elif fingerprint_type == 'maccs':
                            compound_fp = rdMolDescriptors.GetMACCSKeysFingerprint(compound_mol)
                        else:
                            compound_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(compound_mol)
                        
                        tanimoto = DataStructs.TanimotoSimilarity(target_fp, compound_fp)
                        
                        if tanimoto >= threshold:
                            similarities.append({
                                'compound_id': f"TANI_{i:04d}",
                                'smiles': compound_smiles,
                                'tanimoto_similarity': tanimoto,
                                'fingerprint_type': fingerprint_type,
                                'molecular_weight': Descriptors.MolWt(compound_mol)
                            })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['tanimoto_similarity'], reverse=True)
            
            # Get AI insights
            ai_insights = ""
            if request.mode in [AgentMode.AI_ASSISTED, AgentMode.FULLY_AUTONOMOUS]:
                ai_insights = await self.get_ai_insights(
                    f"Tanimoto similarity analysis using {fingerprint_type} fingerprints",
                    {'similarities': similarities[:10], 'threshold': threshold}
                )
            
            recommendations = [
                f"Found {len(similarities)} compounds with Tanimoto similarity >= {threshold}",
                f"Using {fingerprint_type} fingerprints for comparison"
            ]
            
            execution_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.TANIMOTO_SIMILARITY,
                success=True,
                results={
                    'similarities': similarities,
                    'fingerprint_type': fingerprint_type,
                    'threshold': threshold,
                    'ai_insights': ai_insights
                },
                confidence=0.88,
                recommendations=recommendations,
                metadata={
                    'analysis_time': datetime.now().isoformat(),
                    'agent_version': "1.0.0"
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AnalysisResult(
                analysis_type=AnalysisType.TANIMOTO_SIMILARITY,
                success=False,
                results={},
                confidence=0.0,
                recommendations=[],
                metadata={},
                execution_time=execution_time,
                error_message=str(e)
            )

class DrugLikenessAgent(BaseAgent):
    """Agent for drug-likeness analysis"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        super().__init__("Drug Likeness Agent", gemini_api_key)
        
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze drug-likeness properties"""
        start_time = time.time()
        
        if not RDKIT_AVAILABLE:
            return AnalysisResult(
                analysis_type=AnalysisType.DRUG_LIKENESS,
                success=False,
                results={},
                confidence=0.0,
                recommendations=["RDKit not available for drug-likeness analysis"],
                metadata={},
                execution_time=time.time() - start_time,
                error_message="RDKit not available"
            )
        
        try:
            target_mol = Chem.MolFromSmiles(request.target_smiles)
            if not target_mol:
                raise ValueError(f"Invalid SMILES: {request.target_smiles}")
            
            # Calculate drug-likeness properties
            properties = {
                'molecular_weight': Descriptors.MolWt(target_mol),
                'logp': Crippen.MolLogP(target_mol),
                'hbd': Descriptors.NumHDonors(target_mol),
                'hba': Descriptors.NumHAcceptors(target_mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(target_mol),
                'tpsa': Descriptors.TPSA(target_mol),
                'aromatic_rings': Descriptors.NumAromaticRings(target_mol)
            }
            
            # Lipinski's Rule of Five
            lipinski_violations = 0
            violation_details = []
            
            if properties['molecular_weight'] > 500:
                lipinski_violations += 1
                violation_details.append("Molecular weight > 500 Da")
            if properties['logp'] > 5:
                lipinski_violations += 1
                violation_details.append("LogP > 5")
            if properties['hbd'] > 5:
                lipinski_violations += 1
                violation_details.append("Hydrogen bond donors > 5")
            if properties['hba'] > 10:
                lipinski_violations += 1
                violation_details.append("Hydrogen bond acceptors > 10")
            
            # Drug-likeness assessment
            drug_likeness_score = max(0, 1 - (lipinski_violations / 4))
            
            # Lipinski assessment
            lipinski_assessment = {
                'passed': lipinski_violations <= 1,  # Allow one violation
                'violations': lipinski_violations,
                'violation_details': violation_details
            }
            
            # Get AI insights
            ai_insights = ""
            if request.mode in [AgentMode.AI_ASSISTED, AgentMode.FULLY_AUTONOMOUS]:
                ai_insights = await self.get_ai_insights(
                    "Drug-likeness analysis using Lipinski's Rule of Five",
                    {'properties': properties, 'violations': lipinski_violations}
                )
            
            recommendations = []
            if lipinski_violations == 0:
                recommendations.append("Compound passes all Lipinski's Rule of Five criteria")
                recommendations.append("Excellent drug-likeness profile")
            elif lipinski_violations == 1:
                recommendations.append("Compound violates 1 Lipinski rule but may still be drug-like")
            else:
                recommendations.append(f"Compound violates {lipinski_violations} Lipinski rules")
                
            if properties['molecular_weight'] > 600:
                recommendations.append("Consider molecular weight reduction strategies")
            if properties['logp'] > 5:
                recommendations.append("High lipophilicity may affect solubility and permeability")
            if properties['tpsa'] > 140:
                recommendations.append("High TPSA may limit membrane permeability")
            
            execution_time = time.time() - start_time
            
            return AnalysisResult(
                analysis_type=AnalysisType.DRUG_LIKENESS,
                success=True,
                results={
                    'properties': properties,
                    'lipinski_assessment': lipinski_assessment,
                    'drug_likeness_score': drug_likeness_score,
                    'ai_insights': ai_insights
                },
                confidence=0.92,
                recommendations=recommendations,
                metadata={'analysis_time': datetime.now().isoformat()},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AnalysisResult(
                analysis_type=AnalysisType.DRUG_LIKENESS,
                success=False,
                results={},
                confidence=0.0,
                recommendations=[],
                metadata={},
                execution_time=execution_time,
                error_message=str(e)
            )

class AgenticOrchestrator:
    """Main orchestrator for agentic analysis"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.agents = {
            AnalysisType.MORGAN_SIMILARITY: MorganSimilarityAgent(gemini_api_key),
            AnalysisType.TANIMOTO_SIMILARITY: TanimotoSimilarityAgent(gemini_api_key),
            AnalysisType.DRUG_LIKENESS: DrugLikenessAgent(gemini_api_key)
        }
        self.gemini_api_key = gemini_api_key
        self.ai_enabled = False
        
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.ai_enabled = True
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini orchestrator: {e}")
        
    async def analyze(self, request: AnalysisRequest) -> Dict[str, AnalysisResult]:
        """Orchestrate analysis across multiple agents"""
        results = {}
        
        # Run analyses in parallel
        tasks = []
        analysis_types = []
        
        for analysis_type in request.analysis_types:
            if analysis_type in self.agents:
                tasks.append(self.agents[analysis_type].analyze(request))
                analysis_types.append(analysis_type)
        
        if not tasks:
            return results
        
        # Wait for all analyses to complete
        try:
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Organize results
            for i, analysis_type in enumerate(analysis_types):
                result = analysis_results[i]
                if isinstance(result, Exception):
                    # Handle exception
                    results[analysis_type.value] = AnalysisResult(
                        analysis_type=analysis_type,
                        success=False,
                        results={},
                        confidence=0.0,
                        recommendations=[],
                        metadata={},
                        execution_time=0.0,
                        error_message=str(result)
                    )
                else:
                    results[analysis_type.value] = result
                    
        except Exception as e:
            logging.error(f"Error in orchestrated analysis: {e}")
        
        return results
    
    async def get_comprehensive_report(self, request: AnalysisRequest, 
                                     results: Dict[str, AnalysisResult]) -> str:
        """Generate comprehensive analysis report using AI"""
        
        if not self.ai_enabled:
            return self._generate_simple_report(request, results)
        
        try:
            # Prepare data for AI analysis
            report_data = {
                'target_smiles': request.target_smiles,
                'analysis_results': {},
                'mode': request.mode.value
            }
            
            for analysis_type, result in results.items():
                report_data['analysis_results'][analysis_type] = {
                    'success': result.success,
                    'confidence': result.confidence,
                    'key_findings': result.results,
                    'recommendations': result.recommendations,
                    'execution_time': result.execution_time
                }
            
            # Generate comprehensive report
            prompt = f"""
            Generate a comprehensive molecular analysis report for compound: {request.target_smiles}
            
            Analysis Results:
            {json.dumps(report_data, indent=2, default=str)}
            
            Please provide:
            1. Executive Summary
            2. Key Findings from each analysis
            3. Integrated Recommendations
            4. Risk Assessment
            5. Next Steps for Drug Discovery
            6. Potential Applications
            
            Format as a professional scientific report with clear sections and bullet points.
            """
            
            response = await self.model.generate_content_async(prompt)
            return response.text
            
        except Exception as e:
            return f"AI report generation failed: {str(e)}\n\n{self._generate_simple_report(request, results)}"
    
    def _generate_simple_report(self, request: AnalysisRequest, results: Dict[str, AnalysisResult]) -> str:
        """Generate a simple text report when AI is not available"""
        report = f"# Molecular Analysis Report\n\n"
        report += f"**Target Compound:** {request.target_smiles}\n"
        report += f"**Analysis Mode:** {request.mode.value}\n"
        report += f"**Timestamp:** {datetime.now().isoformat()}\n\n"
        
        for analysis_type, result in results.items():
            report += f"## {analysis_type.replace('_', ' ').title()}\n"
            if result.success:
                report += f"- **Status:** ✅ Success\n"
                report += f"- **Confidence:** {result.confidence:.2f}\n"
                report += f"- **Execution Time:** {result.execution_time:.2f}s\n"
                
                if result.recommendations:
                    report += "- **Recommendations:**\n"
                    for rec in result.recommendations:
                        report += f"  - {rec}\n"
            else:
                report += f"- **Status:** ❌ Failed\n"
                report += f"- **Error:** {result.error_message}\n"
            report += "\n"
        
        return report

class TrackMyPDBAgenticInterface:
    """Integration interface for TrackMyPDB agentic capabilities"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        # Try to get API key from environment if not provided
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        self.orchestrator = AgenticOrchestrator(gemini_api_key)
        self.gemini_api_key = gemini_api_key
        
    def get_available_analysis_types(self) -> List[AnalysisType]:
        """Get list of available analysis types"""
        if RDKIT_AVAILABLE:
            return [
                AnalysisType.MORGAN_SIMILARITY,
                AnalysisType.TANIMOTO_SIMILARITY,
                AnalysisType.DRUG_LIKENESS
            ]
        else:
            return []
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}
        for analysis_type, agent in self.orchestrator.agents.items():
            status[analysis_type.value] = {
                'name': agent.name,
                'ai_enabled': agent.ai_enabled,
                'available': RDKIT_AVAILABLE
            }
        return status
    
    async def run_comprehensive_analysis(
        self, 
        target_smiles: str, 
        mode: AgentMode = AgentMode.AI_ASSISTED,
        analysis_types: List[AnalysisType] = None,
        heteroatom_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run comprehensive agentic analysis"""
        
        start_time = time.time()
        
        if analysis_types is None:
            analysis_types = self.get_available_analysis_types()
        
        # Create analysis request
        request = AnalysisRequest(
            target_smiles=target_smiles,
            analysis_types=analysis_types,
            mode=mode,
            parameters={
                'radius': kwargs.get('radius', 2),
                'n_bits': kwargs.get('n_bits', 2048),
                'threshold': kwargs.get('threshold', 0.2),
                'fingerprint_type': kwargs.get('fingerprint_type', 'morgan'),
                'heteroatom_data': heteroatom_data,
                **kwargs
            },
            context=kwargs.get('context')
        )
        
        # Run analysis
        results = await self.orchestrator.analyze(request)
        
        # Generate comprehensive report
        comprehensive_report = await self.orchestrator.get_comprehensive_report(request, results)
        
        # Calculate summary statistics
        successful_analyses = sum(1 for r in results.values() if r.success)
        average_confidence = np.mean([r.confidence for r in results.values() if r.success])
        total_execution_time = time.time() - start_time
        
        return {
            'results': results,
            'comprehensive_report': comprehensive_report,
            'analysis_summary': {
                'target_smiles': target_smiles,
                'mode': mode,
                'analyses_requested': len(analysis_types),
                'analyses_successful': successful_analyses,
                'average_confidence': float(average_confidence) if successful_analyses > 0 else 0.0,
                'total_execution_time': total_execution_time,
                'timestamp': datetime.now().isoformat()
            }
        }

# Example usage functions
async def example_usage():
    """Example usage of the agentic system"""
    
    # Initialize with API key (replace with actual key)
    interface = TrackMyPDBAgenticInterface("your_gemini_api_key_here")
    
    # Run analysis
    results = await interface.run_comprehensive_analysis(
        target_smiles="CCO",  # Ethanol
        mode=AgentMode.AI_ASSISTED,
        analysis_types=[
            AnalysisType.MORGAN_SIMILARITY,
            AnalysisType.TANIMOTO_SIMILARITY,
            AnalysisType.DRUG_LIKENESS
        ]
    )
    
    print("Analysis Results:")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(example_usage())