"""
Autonomous Iterator for TrackMyPDB
@author AI Assistant

Intelligent iteration controller that autonomously processes batches of heteroatom data
with AI guidance, adaptive learning, and intelligent stopping conditions.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from .adaptive_batch_processor import AdaptiveBatchProcessor
from .agentic_layer import TrackMyPDBAgenticInterface, AgentMode, AnalysisType
from .gemini_agent import GeminiAgent
from .performance_monitor import PerformanceMonitor

class IterationMode(Enum):
    """Iteration modes for autonomous processing"""
    CONTINUOUS = "continuous"
    BATCH_BASED = "batch_based"
    TIME_LIMITED = "time_limited"
    CONVERGENCE_BASED = "convergence_based"
    AI_GUIDED = "ai_guided"

class IterationState(Enum):
    """Current state of iteration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class IterationConfig:
    """Configuration for autonomous iteration"""
    mode: IterationMode = IterationMode.AI_GUIDED
    max_iterations: int = 100
    max_time_minutes: int = 60
    convergence_threshold: float = 0.95
    batch_size: int = 50
    min_similarity_threshold: float = 0.3
    enable_ai_guidance: bool = True
    save_progress: bool = True
    auto_adjust_parameters: bool = True
    quality_threshold: float = 0.8

@dataclass
class IterationResult:
    """Results from a single iteration"""
    iteration_number: int
    batch_data: pd.DataFrame
    analysis_results: Dict[str, Any]
    similarity_results: pd.DataFrame
    quality_score: float
    processing_time: float
    ai_insights: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    should_continue: bool = True
    convergence_metrics: Dict[str, float] = field(default_factory=dict)

class AutonomousIterator:
    """
    Autonomous iterator for continuous molecular analysis
    """
    
    def __init__(self, 
                 data_source: pd.DataFrame,
                 agentic_interface: TrackMyPDBAgenticInterface,
                 gemini_agent: Optional[GeminiAgent] = None,
                 config: Optional[IterationConfig] = None):
        
        self.data_source = data_source.copy()
        self.agentic_interface = agentic_interface
        self.gemini_agent = gemini_agent
        self.config = config or IterationConfig()
        
        # State management
        self.state = IterationState.IDLE
        self.current_iteration = 0
        self.start_time = None
        self.results_history: List[IterationResult] = []
        self.processed_indices = set()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        self.batch_processor = AdaptiveBatchProcessor()
        
        # Convergence tracking
        self.quality_scores = []
        self.similarity_distributions = []
        self.convergence_metrics = {}
        
        # Progress saving
        self.progress_file = Path("iteration_progress.json")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def start_autonomous_iteration(self, 
                                       target_smiles: Optional[str] = None,
                                       analysis_types: Optional[List[AnalysisType]] = None) -> Dict[str, Any]:
        """Start autonomous iteration process"""
        
        self.logger.info("🚀 Starting autonomous iteration process...")
        self.state = IterationState.RUNNING
        self.start_time = time.time()
        
        try:
            # Load previous progress if available
            if self.config.save_progress and self.progress_file.exists():
                await self._load_progress()
            
            # Prepare data batches
            data_batches = self._prepare_data_batches()
            
            # Main iteration loop
            while await self._should_continue_iteration():
                iteration_start = time.time()
                
                # Get next batch
                batch_data = await self._get_next_batch(data_batches)
                if batch_data.empty:
                    self.logger.info("📊 No more data to process")
                    break
                
                # Process batch with AI guidance
                iteration_result = await self._process_iteration_batch(
                    batch_data, target_smiles, analysis_types
                )
                
                # Store results and update metrics
                self.results_history.append(iteration_result)
                await self._update_convergence_metrics(iteration_result)
                
                # AI decision on continuation
                if self.config.enable_ai_guidance and self.gemini_agent:
                    should_continue = await self._ai_continuation_decision(iteration_result)
                    if not should_continue:
                        self.logger.info("🤖 AI recommends stopping iteration")
                        break
                
                # Save progress
                if self.config.save_progress:
                    await self._save_progress()
                
                # Adaptive pause between iterations
                await self._adaptive_pause()
                
                self.current_iteration += 1
            
            # Finalize results
            final_results = await self._finalize_iteration_results()
            self.state = IterationState.COMPLETED
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Iteration failed: {str(e)}")
            self.state = IterationState.ERROR
            raise
    
    async def _process_iteration_batch(self, 
                                     batch_data: pd.DataFrame,
                                     target_smiles: Optional[str],
                                     analysis_types: Optional[List[AnalysisType]]) -> IterationResult:
        """Process a single iteration batch"""
        
        iteration_start = time.time()
        
        # Extract SMILES for similarity analysis
        if target_smiles is None and 'SMILES' in batch_data.columns:
            # Use most frequent or highest quality SMILES as target
            target_smiles = self._select_target_smiles(batch_data)
        
        # Prepare analysis parameters
        analysis_types = analysis_types or [
            AnalysisType.MORGAN_SIMILARITY,
            AnalysisType.TANIMOTO_SIMILARITY,
            AnalysisType.DRUG_LIKENESS
        ]
        
        # Run comprehensive analysis
        analysis_results = await self.agentic_interface.run_comprehensive_analysis(
            target_smiles=target_smiles,
            mode=AgentMode.FULLY_AUTONOMOUS,
            analysis_types=analysis_types,
            heteroatom_data=batch_data
        )
        
        # Perform similarity analysis on batch
        similarity_results = await self._analyze_batch_similarity(batch_data, target_smiles)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(analysis_results, similarity_results)
        
        # Generate AI insights if available
        ai_insights = None
        recommendations = []
        
        if self.config.enable_ai_guidance and self.gemini_agent and self.gemini_agent.is_available():
            ai_insights = await self._generate_ai_insights(batch_data, analysis_results, similarity_results)
            recommendations = await self._generate_ai_recommendations(analysis_results, quality_score)
        
        # Determine continuation
        should_continue = await self._evaluate_continuation_criteria(quality_score, similarity_results)
        
        processing_time = time.time() - iteration_start
        
        return IterationResult(
            iteration_number=self.current_iteration,
            batch_data=batch_data,
            analysis_results=analysis_results,
            similarity_results=similarity_results,
            quality_score=quality_score,
            processing_time=processing_time,
            ai_insights=ai_insights,
            recommendations=recommendations,
            should_continue=should_continue,
            convergence_metrics=self.convergence_metrics.copy()
        )
    
    async def _should_continue_iteration(self) -> bool:
        """Determine if iteration should continue based on multiple criteria"""
        
        # Check time limit
        if self.start_time and self.config.max_time_minutes > 0:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.config.max_time_minutes:
                self.logger.info(f"⏰ Time limit reached: {elapsed_minutes:.1f} minutes")
                return False
        
        # Check iteration limit
        if self.current_iteration >= self.config.max_iterations:
            self.logger.info(f"🔢 Iteration limit reached: {self.current_iteration}")
            return False
        
        # Check convergence
        if len(self.quality_scores) >= 5:
            recent_scores = self.quality_scores[-5:]
            convergence = np.std(recent_scores) < 0.05  # Low variance indicates convergence
            if convergence and np.mean(recent_scores) >= self.config.convergence_threshold:
                self.logger.info(f"📈 Convergence achieved: {np.mean(recent_scores):.3f}")
                return False
        
        # Check if we have unprocessed data
        remaining_data = len(self.data_source) - len(self.processed_indices)
        if remaining_data == 0:
            self.logger.info("📊 All data processed")
            return False
        
        # Check system resources
        if self.batch_processor.should_pause_processing():
            self.logger.info("💻 System resources constrained, pausing")
            self.state = IterationState.PAUSED
            await asyncio.sleep(30)  # Wait for resources
            self.state = IterationState.RUNNING
        
        return True
    
    async def _ai_continuation_decision(self, iteration_result: IterationResult) -> bool:
        """Use AI to decide whether to continue iteration"""
        
        if not self.gemini_agent or not self.gemini_agent.is_available():
            return True
        
        try:
            # Prepare context for AI decision
            context = {
                'current_iteration': self.current_iteration,
                'quality_score': iteration_result.quality_score,
                'processing_time': iteration_result.processing_time,
                'similarity_stats': {
                    'max_similarity': iteration_result.similarity_results['Tanimoto_Similarity'].max() 
                    if not iteration_result.similarity_results.empty and 'Tanimoto_Similarity' in iteration_result.similarity_results.columns else 0,
                    'mean_similarity': iteration_result.similarity_results['Tanimoto_Similarity'].mean()
                    if not iteration_result.similarity_results.empty and 'Tanimoto_Similarity' in iteration_result.similarity_results.columns else 0,
                    'count': len(iteration_result.similarity_results)
                },
                'convergence_trend': self.quality_scores[-5:] if len(self.quality_scores) >= 5 else self.quality_scores,
                'remaining_data': len(self.data_source) - len(self.processed_indices)
            }
            
            return self.gemini_agent.should_continue_iteration(context)
            
        except Exception as e:
            self.logger.warning(f"AI continuation decision failed: {e}")
            return True  # Conservative default
    
    def _prepare_data_batches(self) -> List[pd.DataFrame]:
        """Prepare data into intelligent batches"""
        
        # Remove already processed data
        unprocessed_data = self.data_source.drop(index=list(self.processed_indices))
        
        if unprocessed_data.empty:
            return []
        
        # Adaptive batch sizing based on data characteristics
        batch_size = self.config.batch_size
        
        # Adjust batch size based on data complexity
        if 'SMILES' in unprocessed_data.columns:
            avg_smiles_length = unprocessed_data['SMILES'].str.len().mean()
            if avg_smiles_length > 100:  # Complex molecules
                batch_size = max(10, batch_size // 2)
        
        # Create batches
        batches = []
        for i in range(0, len(unprocessed_data), batch_size):
            batch = unprocessed_data.iloc[i:i + batch_size].copy()
            batches.append(batch)
        
        self.logger.info(f"📦 Prepared {len(batches)} batches for processing")
        return batches
    
    async def _get_next_batch(self, data_batches: List[pd.DataFrame]) -> pd.DataFrame:
        """Get next batch for processing with intelligent selection"""
        
        if self.current_iteration >= len(data_batches):
            return pd.DataFrame()
        
        batch = data_batches[self.current_iteration]
        
        # Mark indices as processed
        self.processed_indices.update(batch.index.tolist())
        
        return batch
    
    async def _analyze_batch_similarity(self, 
                                      batch_data: pd.DataFrame, 
                                      target_smiles: str) -> pd.DataFrame:
        """Analyze molecular similarity within batch"""
        
        if 'SMILES' not in batch_data.columns or not target_smiles:
            return pd.DataFrame()
        
        try:
            # Use the existing similarity analyzer
            from .advanced_similarity_analyzer import AdvancedSimilarityAnalyzer
            
            analyzer = AdvancedSimilarityAnalyzer()
            
            # Prepare compound list
            compounds = []
            for _, row in batch_data.iterrows():
                if pd.notna(row.get('SMILES')):
                    compounds.append({
                        'smiles': row['SMILES'],
                        'pdb_id': row.get('PDB_ID', ''),
                        'heteroatom_code': row.get('Heteroatom_Code', ''),
                        'chemical_name': row.get('Chemical_Name', '')
                    })
            
            if not compounds:
                return pd.DataFrame()
            
            # Analyze similarity
            results = analyzer.analyze_similarity_batch(
                target_smiles=target_smiles,
                compound_list=compounds,
                methods=['morgan'],
                similarity_metrics=['tanimoto'],
                min_similarity=self.config.min_similarity_threshold
            )
            
            # Convert to DataFrame
            similarity_data = []
            for method, similarity_results in results.items():
                for result in similarity_results:
                    similarity_data.append({
                        'PDB_ID': result.compound_data.get('pdb_id', ''),
                        'Heteroatom_Code': result.compound_data.get('heteroatom_code', ''),
                        'SMILES': result.compound_data.get('smiles', ''),
                        'Chemical_Name': result.compound_data.get('chemical_name', ''),
                        'Tanimoto_Similarity': result.similarity_score,
                        'Method': method
                    })
            
            return pd.DataFrame(similarity_data)
            
        except Exception as e:
            self.logger.error(f"Similarity analysis failed: {e}")
            return pd.DataFrame()
    
    def _calculate_quality_score(self, 
                               analysis_results: Dict[str, Any], 
                               similarity_results: pd.DataFrame) -> float:
        """Calculate overall quality score for the iteration"""
        
        scores = []
        
        # Analysis success rate
        if 'analysis_summary' in analysis_results:
            summary = analysis_results['analysis_summary']
            success_rate = summary.get('analyses_successful', 0) / max(summary.get('analyses_requested', 1), 1)
            scores.append(success_rate)
        
        # Similarity distribution quality
        if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns:
            similarities = similarity_results['Tanimoto_Similarity']
            # Prefer diverse similarity distribution
            diversity = 1 - similarities.std() if similarities.std() > 0 else 0.5
            mean_similarity = similarities.mean()
            similarity_quality = (mean_similarity + diversity) / 2
            scores.append(similarity_quality)
        
        # Data completeness
        if not similarity_results.empty:
            completeness = len(similarity_results) / max(len(similarity_results), 1)
            scores.append(completeness)
        
        # Overall quality
        quality_score = np.mean(scores) if scores else 0.0
        self.quality_scores.append(quality_score)
        
        return quality_score
    
    async def _generate_ai_insights(self, 
                                  batch_data: pd.DataFrame,
                                  analysis_results: Dict[str, Any],
                                  similarity_results: pd.DataFrame) -> str:
        """Generate AI insights for the current iteration"""
        
        if not self.gemini_agent or not self.gemini_agent.is_available():
            return "AI insights not available"
        
        try:
            context = {
                'batch_size': len(batch_data),
                'analysis_results': analysis_results.get('analysis_summary', {}),
                'similarity_count': len(similarity_results),
                'iteration_number': self.current_iteration,
                'quality_trend': self.quality_scores[-3:] if len(self.quality_scores) >= 3 else self.quality_scores
            }
            
            prompt = f"""
            Analyze this molecular analysis iteration and provide scientific insights:
            
            Iteration Context: {context}
            
            Provide insights on:
            1. Quality of current batch analysis
            2. Trends observed in molecular similarities
            3. Recommendations for parameter optimization
            4. Scientific significance of findings
            
            Keep response concise and scientific.
            """
            
            return self.gemini_agent.generate_ai_response_sync(prompt)
            
        except Exception as e:
            self.logger.error(f"AI insights generation failed: {e}")
            return "AI insights generation encountered an error"
    
    async def _generate_ai_recommendations(self, 
                                         analysis_results: Dict[str, Any],
                                         quality_score: float) -> List[str]:
        """Generate AI recommendations for optimization"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.6:
            recommendations.append("Consider adjusting similarity thresholds")
            recommendations.append("Increase batch size for better statistics")
        elif quality_score > 0.9:
            recommendations.append("Excellent quality - consider expanding analysis scope")
        
        # Analysis-specific recommendations
        if 'analysis_summary' in analysis_results:
            summary = analysis_results['analysis_summary']
            if summary.get('average_confidence', 0) < 0.7:
                recommendations.append("Low confidence detected - verify input data quality")
        
        return recommendations
    
    async def _evaluate_continuation_criteria(self, 
                                            quality_score: float,
                                            similarity_results: pd.DataFrame) -> bool:
        """Evaluate if iteration should continue based on results"""
        
        # Stop if quality is consistently low
        if len(self.quality_scores) >= 3:
            recent_scores = self.quality_scores[-3:]
            if all(score < 0.5 for score in recent_scores):
                return False
        
        # Stop if no meaningful similarities found
        if similarity_results.empty:
            return False
        
        # Continue if finding good results
        if not similarity_results.empty and 'Tanimoto_Similarity' in similarity_results.columns:
            max_similarity = similarity_results['Tanimoto_Similarity'].max()
            if max_similarity > 0.8:  # High similarity found
                return True
        
        return True
    
    async def _update_convergence_metrics(self, iteration_result: IterationResult):
        """Update convergence tracking metrics"""
        
        # Quality convergence
        self.convergence_metrics['quality_variance'] = np.var(self.quality_scores) if len(self.quality_scores) > 1 else 1.0
        self.convergence_metrics['quality_trend'] = np.mean(np.diff(self.quality_scores)) if len(self.quality_scores) > 1 else 0.0
        
        # Similarity distribution convergence
        if not iteration_result.similarity_results.empty and 'Tanimoto_Similarity' in iteration_result.similarity_results.columns:
            similarities = iteration_result.similarity_results['Tanimoto_Similarity'].values
            self.similarity_distributions.append(similarities)
            
            if len(self.similarity_distributions) > 1:
                # Compare with previous distribution
                prev_dist = self.similarity_distributions[-2]
                curr_dist = similarities
                
                # Statistical similarity between distributions
                from scipy import stats
                try:
                    ks_stat, p_value = stats.ks_2samp(prev_dist, curr_dist)
                    self.convergence_metrics['distribution_similarity'] = 1 - ks_stat
                except:
                    self.convergence_metrics['distribution_similarity'] = 0.5
    
    async def _adaptive_pause(self):
        """Implement adaptive pause between iterations"""
        
        # Base pause time
        pause_time = 1.0
        
        # Adjust based on system load
        if self.batch_processor.should_pause_processing():
            pause_time = 5.0
        
        # Adjust based on quality trends
        if len(self.quality_scores) >= 2:
            if self.quality_scores[-1] < self.quality_scores[-2]:
                pause_time += 2.0  # Longer pause if quality declining
        
        await asyncio.sleep(pause_time)
    
    async def _save_progress(self):
        """Save iteration progress to file"""
        
        try:
            progress_data = {
                'current_iteration': self.current_iteration,
                'processed_indices': list(self.processed_indices),
                'quality_scores': self.quality_scores,
                'convergence_metrics': self.convergence_metrics,
                'config': {
                    'mode': self.config.mode.value,
                    'max_iterations': self.config.max_iterations,
                    'batch_size': self.config.batch_size,
                    'min_similarity_threshold': self.config.min_similarity_threshold
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    async def _load_progress(self):
        """Load previous iteration progress"""
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            self.current_iteration = progress_data.get('current_iteration', 0)
            self.processed_indices = set(progress_data.get('processed_indices', []))
            self.quality_scores = progress_data.get('quality_scores', [])
            self.convergence_metrics = progress_data.get('convergence_metrics', {})
            
            self.logger.info(f"📂 Loaded progress: iteration {self.current_iteration}")
            
        except Exception as e:
            self.logger.error(f"Failed to load progress: {e}")
    
    async def _finalize_iteration_results(self) -> Dict[str, Any]:
        """Finalize and summarize all iteration results"""
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Aggregate results
        all_similarity_results = pd.concat([
            result.similarity_results for result in self.results_history
            if not result.similarity_results.empty
        ], ignore_index=True)
        
        # Calculate summary statistics
        summary_stats = {
            'total_iterations': len(self.results_history),
            'total_processing_time': total_time,
            'average_quality_score': np.mean(self.quality_scores) if self.quality_scores else 0,
            'final_convergence_metrics': self.convergence_metrics,
            'total_compounds_analyzed': len(self.processed_indices),
            'total_similarities_found': len(all_similarity_results),
            'best_similarity': all_similarity_results['Tanimoto_Similarity'].max() 
            if not all_similarity_results.empty and 'Tanimoto_Similarity' in all_similarity_results.columns else 0
        }
        
        # Generate final AI summary
        final_ai_summary = ""
        if self.gemini_agent and self.gemini_agent.is_available():
            try:
                final_ai_summary = self.gemini_agent.generate_ai_response_sync(f"""
                Provide a comprehensive summary of this autonomous iteration analysis:
                
                Summary Statistics: {summary_stats}
                Quality Trend: {self.quality_scores}
                
                Summarize:
                1. Overall analysis performance and quality
                2. Key findings and patterns discovered
                3. Recommendations for future analysis
                4. Scientific significance of results
                """)
            except Exception as e:
                final_ai_summary = f"AI summary generation failed: {e}"
        
        return {
            'summary_statistics': summary_stats,
            'all_similarity_results': all_similarity_results,
            'quality_history': self.quality_scores,
            'convergence_metrics': self.convergence_metrics,
            'iteration_results': [
                {
                    'iteration': result.iteration_number,
                    'quality_score': result.quality_score,
                    'processing_time': result.processing_time,
                    'compounds_processed': len(result.batch_data),
                    'similarities_found': len(result.similarity_results),
                    'ai_insights': result.ai_insights
                }
                for result in self.results_history
            ],
            'final_ai_summary': final_ai_summary,
            'completion_status': 'success',
            'state': self.state.value
        }
    
    def _select_target_smiles(self, batch_data: pd.DataFrame) -> str:
        """Select the best target SMILES from batch data"""
        
        if 'SMILES' not in batch_data.columns:
            return ""
        
        valid_smiles = batch_data.dropna(subset=['SMILES'])
        if valid_smiles.empty:
            return ""
        
        # Prefer molecules with moderate complexity (not too simple, not too complex)
        smiles_lengths = valid_smiles['SMILES'].str.len()
        median_length = smiles_lengths.median()
        
        # Find SMILES closest to median length
        closest_idx = (smiles_lengths - median_length).abs().idxmin()
        return valid_smiles.loc[closest_idx, 'SMILES']
    
    def get_iteration_status(self) -> Dict[str, Any]:
        """Get current iteration status"""
        
        return {
            'state': self.state.value,
            'current_iteration': self.current_iteration,
            'total_processed': len(self.processed_indices),
            'quality_scores': self.quality_scores,
            'convergence_metrics': self.convergence_metrics,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
    
    async def pause_iteration(self):
        """Pause the iteration process"""
        self.state = IterationState.PAUSED
        self.logger.info("⏸️ Iteration paused")
    
    async def resume_iteration(self):
        """Resume the iteration process"""
        if self.state == IterationState.PAUSED:
            self.state = IterationState.RUNNING
            self.logger.info("▶️ Iteration resumed")
    
    async def stop_iteration(self):
        """Stop the iteration process"""
        self.state = IterationState.COMPLETED
        if self.config.save_progress:
            await self._save_progress()
        self.logger.info("⏹️ Iteration stopped")

# Utility functions for easy integration
async def create_autonomous_iterator(data_source: pd.DataFrame,
                                   agentic_interface: TrackMyPDBAgenticInterface,
                                   gemini_agent: Optional[GeminiAgent] = None,
                                   **config_kwargs) -> AutonomousIterator:
    """Create and configure an autonomous iterator"""
    
    config = IterationConfig(**config_kwargs)
    return AutonomousIterator(
        data_source=data_source,
        agentic_interface=agentic_interface,
        gemini_agent=gemini_agent,
        config=config
    )

async def run_autonomous_analysis(data_file_path: str,
                                target_smiles: Optional[str] = None,
                                max_iterations: int = 50,
                                time_limit_minutes: int = 30) -> Dict[str, Any]:
    """
    High-level function to run autonomous analysis on a CSV file
    """
    
    # Load data
    data = pd.read_csv(data_file_path)
    
    # Create components
    from .agentic_layer import TrackMyPDBAgenticInterface
    from .gemini_agent import GeminiAgent
    
    agentic_interface = TrackMyPDBAgenticInterface()
    gemini_agent = GeminiAgent()
    
    # Create iterator
    iterator = await create_autonomous_iterator(
        data_source=data,
        agentic_interface=agentic_interface,
        gemini_agent=gemini_agent,
        max_iterations=max_iterations,
        max_time_minutes=time_limit_minutes,
        mode=IterationMode.AI_GUIDED
    )
    
    # Run analysis
    results = await iterator.start_autonomous_iteration(target_smiles=target_smiles)
    
    return results