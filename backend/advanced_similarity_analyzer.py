"""
TrackMyPDB Advanced Similarity Analyzer
@author Anu Gamage

Enhanced similarity analyzer that supports multiple fingerprint types (Morgan, MACCS, AtomPair), 
parallel processing, and adaptive threshold selection based on dataset characteristics.
Licensed under MIT License - Open Source Project
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, AllChem, MACCSkeys, AtomPairs
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, DiceSimilarity, CosineSimilarity
import streamlit as st
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

# Import our caching and monitoring systems
from .molecular_cache import MolecularDataCache
from .performance_monitor import PerformanceMonitor, track_performance, SmartProgressTracker

class FingerprintType(Enum):
    MORGAN = "morgan"
    MACCS = "maccs"
    ATOMPAIR = "atompair"
    TOPOLOGICAL = "topological"
    RDKIT = "rdkit"
    AVALON = "avalon"
    PATTERN = "pattern"

class SimilarityMetric(Enum):
    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    SOKAL = "sokal"
    KULCZYNSKI = "kulczynski"

@dataclass
class FingerprintConfig:
    fp_type: FingerprintType
    radius: int = 2
    n_bits: int = 2048
    use_features: bool = False
    use_chirality: bool = False

@dataclass
class SimilarityResult:
    compound_id: str
    smiles: str
    similarity_score: float
    fingerprint_type: str
    metric_type: str
    pdb_id: str
    heteroatom_code: str
    chemical_name: str
    formula: str
    molecular_weight: float
    additional_properties: Dict[str, Any]

class AdvancedSimilarityAnalyzer:
    """
    Enhanced similarity analyzer with multiple fingerprint types and parallel processing
    """
    
    def __init__(self, 
                 cache: Optional[MolecularDataCache] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 progress_tracker: Optional[SmartProgressTracker] = None,
                 max_workers: Optional[int] = None):
        
        self.cache = cache
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.progress_tracker = progress_tracker or SmartProgressTracker()
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        # Fingerprint generators with enhanced options
        self.fingerprint_generators = {
            FingerprintType.MORGAN: self._generate_morgan,
            FingerprintType.MACCS: self._generate_maccs,
            FingerprintType.ATOMPAIR: self._generate_atompair,
            FingerprintType.TOPOLOGICAL: self._generate_topological,
            FingerprintType.RDKIT: self._generate_rdkit,
            FingerprintType.AVALON: self._generate_avalon,
            FingerprintType.PATTERN: self._generate_pattern
        }
        
        # Similarity metrics
        self.similarity_metrics = {
            SimilarityMetric.TANIMOTO: TanimotoSimilarity,
            SimilarityMetric.DICE: DiceSimilarity,
            SimilarityMetric.COSINE: CosineSimilarity,
            SimilarityMetric.SOKAL: self._sokal_similarity,
            SimilarityMetric.KULCZYNSKI: self._kulczynski_similarity
        }
        
        # Cache for computed fingerprints
        self.fingerprint_cache = {}
        self.molecule_cache = {}
    
    def _generate_morgan(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate Morgan fingerprint"""
        try:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 
                radius=config.radius, 
                nBits=config.n_bits,
                useFeatures=config.use_features,
                useChirality=config.use_chirality
            )
        except Exception as e:
            logging.error(f"Error generating Morgan fingerprint: {e}")
            return None
    
    def _generate_maccs(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate MACCS keys fingerprint"""
        try:
            return MACCSkeys.GenMACCSKeys(mol)
        except Exception as e:
            logging.error(f"Error generating MACCS fingerprint: {e}")
            return None
    
    def _generate_atompair(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate AtomPair fingerprint"""
        try:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=config.n_bits
            )
        except Exception as e:
            logging.error(f"Error generating AtomPair fingerprint: {e}")
            return None
    
    def _generate_topological(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate topological torsion fingerprint"""
        try:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=config.n_bits
            )
        except Exception as e:
            logging.error(f"Error generating topological fingerprint: {e}")
            return None
    
    def _generate_rdkit(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate RDKit fingerprint"""
        try:
            return Chem.RDKFingerprint(mol, fpSize=config.n_bits)
        except Exception as e:
            logging.error(f"Error generating RDKit fingerprint: {e}")
            return None
    
    def _generate_avalon(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate Avalon fingerprint"""
        try:
            from rdkit.Avalon import pyAvalonTools
            return pyAvalonTools.GetAvalonFP(mol, config.n_bits)
        except ImportError:
            st.warning("Avalon fingerprints not available - install rdkit-pypi with Avalon support")
            return None
        except Exception as e:
            logging.error(f"Error generating Avalon fingerprint: {e}")
            return None
    
    def _generate_pattern(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate pattern fingerprint"""
        try:
            return Chem.PatternFingerprint(mol, fpSize=config.n_bits)
        except Exception as e:
            logging.error(f"Error generating pattern fingerprint: {e}")
            return None
    
    def _sokal_similarity(self, fp1: Any, fp2: Any) -> float:
        """Calculate Sokal similarity"""
        try:
            return DataStructs.SokalSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def _kulczynski_similarity(self, fp1: Any, fp2: Any) -> float:
        """Calculate Kulczynski similarity"""
        try:
            return DataStructs.KulczynskiSimilarity(fp1, fp2)
        except:
            return 0.0
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule with caching"""
        if not smiles or pd.isna(smiles):
            return None
            
        smiles = smiles.strip()
        if smiles in self.molecule_cache:
            return self.molecule_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Sanitize molecule
                Chem.SanitizeMol(mol)
                self.molecule_cache[smiles] = mol
            return mol
        except Exception as e:
            logging.error(f"Error parsing SMILES '{smiles}': {e}")
            return None
    
    def generate_fingerprint(self, mol: Chem.Mol, config: FingerprintConfig) -> Optional[Any]:
        """Generate fingerprint using specified configuration"""
        if mol is None:
            return None
        
        # Create cache key
        mol_key = Chem.MolToSmiles(mol)
        cache_key = f"{mol_key}_{config.fp_type.value}_{config.radius}_{config.n_bits}_{config.use_features}_{config.use_chirality}"
        
        # Check cache first
        if self.cache:
            cached_fp = self.cache.get_cached_chemical_data(cache_key)
            if cached_fp is not None:
                return cached_fp
        
        # Generate fingerprint
        generator = self.fingerprint_generators.get(config.fp_type)
        if generator is None:
            logging.error(f"Unknown fingerprint type: {config.fp_type}")
            return None
        
        fingerprint = generator(mol, config)
        
        # Cache result
        if fingerprint is not None and self.cache:
            self.cache.store_chemical_data(cache_key, fingerprint)
        
        return fingerprint
    
    def calculate_molecular_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive molecular properties"""
        if mol is None:
            return {}
        
        try:
            return {
                'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'rings': rdMolDescriptors.CalcNumRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'fraction_csp3': rdMolDescriptors.CalcFractionCsp3(mol),
                'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol)
            }
        except Exception as e:
            logging.error(f"Error calculating molecular properties: {e}")
            return {}
    
    @performance_monitor.monitor_analysis_performance
    def process_compound_batch(self, 
                              compounds: List[Dict[str, Any]], 
                              fingerprint_configs: List[FingerprintConfig]) -> List[Dict[str, Any]]:
        """Process a batch of compounds with multiple fingerprint types"""
        results = []
        
        for compound in compounds:
            smiles = compound.get('smiles', '')
            mol = self.smiles_to_mol(smiles)
            
            if mol is None:
                continue
            
            # Calculate molecular properties
            properties = self.calculate_molecular_properties(mol)
            
            # Generate fingerprints for each configuration
            fingerprints = {}
            for config in fingerprint_configs:
                fp = self.generate_fingerprint(mol, config)
                if fp is not None:
                    fingerprints[config.fp_type.value] = fp
            
            if fingerprints:  # Only include compounds with at least one valid fingerprint
                compound_result = {
                    **compound,
                    'mol': mol,
                    'fingerprints': fingerprints,
                    'properties': properties
                }
                results.append(compound_result)
        
        return results
    
    def analyze_similarity_batch(self, 
                                target_smiles: str, 
                                compound_list: List[Dict[str, Any]], 
                                methods: List[str] = ['morgan'],
                                similarity_metrics: List[str] = ['tanimoto'],
                                min_similarity: float = 0.0,
                                top_n: int = 100) -> Dict[str, List[SimilarityResult]]:
        """
        Analyze similarity using multiple methods and metrics with parallel processing
        
        Args:
            target_smiles: Target molecule SMILES
            compound_list: List of compounds to compare
            methods: List of fingerprint methods to use
            similarity_metrics: List of similarity metrics to use
            min_similarity: Minimum similarity threshold
            top_n: Number of top results to return per method
            
        Returns:
            Dictionary with results for each method/metric combination
        """
        with track_performance(f"Similarity Analysis - {len(compound_list)} compounds", 
                             self.progress_tracker, 
                             len(compound_list)) as task:
            
            # Parse target molecule
            target_mol = self.smiles_to_mol(target_smiles)
            if target_mol is None:
                raise ValueError(f"Invalid target SMILES: {target_smiles}")
            
            # Create fingerprint configurations
            configs = []
            for method in methods:
                try:
                    fp_type = FingerprintType(method.lower())
                    config = FingerprintConfig(fp_type=fp_type)
                    configs.append(config)
                except ValueError:
                    st.warning(f"Unknown fingerprint method: {method}")
                    continue
            
            if not configs:
                raise ValueError("No valid fingerprint methods specified")
            
            # Generate target fingerprints
            target_fingerprints = {}
            for config in configs:
                fp = self.generate_fingerprint(target_mol, config)
                if fp is not None:
                    target_fingerprints[config.fp_type.value] = fp
            
            # Process compounds in parallel batches
            batch_size = max(1, len(compound_list) // self.max_workers)
            batches = [compound_list[i:i + batch_size] 
                      for i in range(0, len(compound_list), batch_size)]
            
            all_processed_compounds = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batch processing tasks
                future_to_batch = {
                    executor.submit(self.process_compound_batch, batch, configs): batch 
                    for batch in batches
                }
                
                completed_batches = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        all_processed_compounds.extend(batch_results)
                        completed_batches += 1
                        
                        # Update progress
                        processed_items = completed_batches * batch_size
                        self.progress_tracker.update_progress(
                            task.task_id, 
                            min(processed_items, len(compound_list)),
                            f"Processed {completed_batches}/{len(batches)} batches"
                        )
                        
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
            
            # Calculate similarities
            results = defaultdict(list)
            
            for metric_name in similarity_metrics:
                try:
                    metric = SimilarityMetric(metric_name.lower())
                except ValueError:
                    st.warning(f"Unknown similarity metric: {metric_name}")
                    continue
                
                similarity_func = self.similarity_metrics[metric]
                
                for fp_type_name, target_fp in target_fingerprints.items():
                    method_results = []
                    
                    for compound in all_processed_compounds:
                        if fp_type_name not in compound['fingerprints']:
                            continue
                        
                        compound_fp = compound['fingerprints'][fp_type_name]
                        similarity = similarity_func(target_fp, compound_fp)
                        
                        if similarity >= min_similarity:
                            result = SimilarityResult(
                                compound_id=compound.get('compound_id', ''),
                                smiles=compound.get('smiles', ''),
                                similarity_score=similarity,
                                fingerprint_type=fp_type_name,
                                metric_type=metric_name,
                                pdb_id=compound.get('pdb_id', ''),
                                heteroatom_code=compound.get('heteroatom_code', ''),
                                chemical_name=compound.get('chemical_name', ''),
                                formula=compound.get('formula', ''),
                                molecular_weight=compound['properties'].get('molecular_weight', 0.0),
                                additional_properties=compound['properties']
                            )
                            method_results.append(result)
                    
                    # Sort by similarity and take top N
                    method_results.sort(key=lambda x: x.similarity_score, reverse=True)
                    results[f"{fp_type_name}_{metric_name}"] = method_results[:top_n]
            
            return dict(results)
    
    def adaptive_threshold_selection(self, similarities: List[float], 
                                   dataset_size: int,
                                   target_count: int = 100) -> float:
        """
        Adaptively select similarity threshold based on dataset characteristics
        
        Args:
            similarities: List of similarity scores
            dataset_size: Size of the dataset
            target_count: Target number of results to return
            
        Returns:
            Optimal similarity threshold
        """
        if not similarities:
            return 0.0
        
        similarities = sorted(similarities, reverse=True)
        
        # If we have fewer similarities than target, return minimum
        if len(similarities) <= target_count:
            return min(similarities)
        
        # Use the similarity score at target_count position
        threshold = similarities[target_count - 1]
        
        # Ensure minimum threshold based on dataset size
        if dataset_size > 10000:
            min_threshold = 0.3  # Stricter for large datasets
        elif dataset_size > 1000:
            min_threshold = 0.2
        else:
            min_threshold = 0.1  # More lenient for small datasets
        
        return max(threshold, min_threshold)
    
    def get_similarity_statistics(self, results: Dict[str, List[SimilarityResult]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for similarity results"""
        stats = {}
        
        for method_key, result_list in results.items():
            if not result_list:
                continue
            
            similarities = [r.similarity_score for r in result_list]
            
            stats[method_key] = {
                'count': len(similarities),
                'mean': np.mean(similarities),
                'median': np.median(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'q25': np.percentile(similarities, 25),
                'q75': np.percentile(similarities, 75),
                'unique_pdbs': len(set(r.pdb_id for r in result_list)),
                'unique_ligands': len(set(r.heteroatom_code for r in result_list))
            }
        
        return stats
    
    def export_results_to_dataframe(self, results: Dict[str, List[SimilarityResult]]) -> pd.DataFrame:
        """Export similarity results to a pandas DataFrame"""
        all_results = []
        
        for method_key, result_list in results.items():
            for result in result_list:
                row = {
                    'method': method_key,
                    'compound_id': result.compound_id,
                    'smiles': result.smiles,
                    'similarity_score': result.similarity_score,
                    'fingerprint_type': result.fingerprint_type,
                    'metric_type': result.metric_type,
                    'pdb_id': result.pdb_id,
                    'heteroatom_code': result.heteroatom_code,
                    'chemical_name': result.chemical_name,
                    'formula': result.formula,
                    'molecular_weight': result.molecular_weight,
                    **{f'prop_{k}': v for k, v in result.additional_properties.items()}
                }
                all_results.append(row)
        
        return pd.DataFrame(all_results)

# Convenience functions for easy integration

def create_advanced_analyzer(use_cache: bool = True, 
                           use_monitoring: bool = True) -> AdvancedSimilarityAnalyzer:
    """Create an advanced similarity analyzer with default configurations"""
    cache = None
    if use_cache:
        from .molecular_cache import create_default_cache
        cache = create_default_cache()
    
    monitor = PerformanceMonitor() if use_monitoring else None
    tracker = SmartProgressTracker() if use_monitoring else None
    
    return AdvancedSimilarityAnalyzer(
        cache=cache,
        performance_monitor=monitor,
        progress_tracker=tracker
    )

async def analyze_similarity_async(target_smiles: str,
                                 compounds: List[Dict[str, Any]],
                                 methods: List[str] = ['morgan', 'maccs'],
                                 metrics: List[str] = ['tanimoto'],
                                 min_similarity: float = 0.1) -> Dict[str, List[SimilarityResult]]:
    """Async wrapper for similarity analysis"""
    analyzer = create_advanced_analyzer()
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            analyzer.analyze_similarity_batch,
            target_smiles,
            compounds,
            methods,
            metrics,
            min_similarity
        )
    
    return results