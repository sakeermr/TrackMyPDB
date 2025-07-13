"""
Comprehensive Database Loader for TrackMyPDB
Connects the massive heteroatom CSV database to AI-assisted and Fully Autonomous modes
@author AI Assistant
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseStats:
    """Statistics about the loaded database"""
    total_compounds: int
    total_pdbs: int
    compounds_with_smiles: int
    unique_heteroatom_codes: int
    average_atom_count: float
    status_distribution: Dict[str, int]
    formula_distribution: Dict[str, int]

class ComprehensiveHeteroatomDatabase:
    """
    Comprehensive loader and manager for the massive heteroatom CSV database
    Connects all batch files and pipeline data for AI-assisted and Fully Autonomous modes
    """
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.database_df: Optional[pd.DataFrame] = None
        self.stats: Optional[DatabaseStats] = None
        self.sqlite_db_path = self.data_folder / "heteroatom_database.sqlite"
        self.cache_enabled = True
        self.index_columns = ['PDB_ID', 'Heteroatom_Code', 'SMILES', 'Chemical_Name']
        
        logger.info(f"Initialized ComprehensiveHeteroatomDatabase with data folder: {self.data_folder}")
    
    def discover_csv_files(self) -> List[Path]:
        """Discover all CSV files in the data directory"""
        csv_files = []
        
        # Find all CSV files matching common patterns
        patterns = [
            "pdb_heteroatoms_batch_*.csv",
            "complete_pipeline_*.csv",
            "Het-*.csv",
            "heteroatoms_*.csv"
        ]
        
        for pattern in patterns:
            files = list(self.data_folder.glob(pattern))
            csv_files.extend(files)
        
        # Also check for any other CSV files
        other_csvs = [f for f in self.data_folder.glob("*.csv") if f not in csv_files]
        csv_files.extend(other_csvs)
        
        logger.info(f"Discovered {len(csv_files)} CSV files")
        return sorted(csv_files)
    
    def load_single_csv(self, file_path: Path) -> pd.DataFrame:
        """Load a single CSV file with error handling"""
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {file_path.name}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def load_comprehensive_database(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the comprehensive heteroatom database from all CSV files
        Combines all batch files and pipeline data into a unified dataset
        """
        
        # Check if cached database exists
        if not force_reload and self.sqlite_db_path.exists():
            logger.info("Loading database from SQLite cache...")
            return self._load_from_sqlite()
        
        logger.info("Loading comprehensive heteroatom database from CSV files...")
        start_time = time.time()
        
        # Discover all CSV files
        csv_files = self.discover_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found in data directory")
            return pd.DataFrame()
        
        # Load all CSV files in parallel
        all_dataframes = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all file loading tasks
            future_to_file = {
                executor.submit(self.load_single_csv, file_path): file_path 
                for file_path in csv_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df = future.result()
                    if not df.empty:
                        # Add source file information
                        df['source_file'] = file_path.name
                        all_dataframes.append(df)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        if not all_dataframes:
            logger.error("No valid CSV files were loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        logger.info(f"Combining {len(all_dataframes)} dataframes...")
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Standardize columns
        combined_df = self._standardize_columns(combined_df)
        
        # Clean and validate data
        combined_df = self._clean_and_validate_data(combined_df)
        
        # Remove duplicates
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['PDB_ID', 'Heteroatom_Code', 'SMILES'], 
            keep='first'
        )
        deduplicated_count = len(combined_df)
        
        logger.info(f"Removed {original_count - deduplicated_count} duplicate entries")
        
        # Create indexes for faster searching
        combined_df = self._create_search_indexes(combined_df)
        
        # Cache to SQLite for faster future loading
        if self.cache_enabled:
            self._save_to_sqlite(combined_df)
        
        # Calculate statistics
        self.stats = self._calculate_statistics(combined_df)
        
        # Store in instance
        self.database_df = combined_df
        
        load_time = time.time() - start_time
        logger.info(f"Successfully loaded comprehensive database: {len(combined_df):,} compounds in {load_time:.2f}s")
        
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and ensure required columns exist"""
        
        # Define column mappings for different file formats
        column_mappings = {
            'pdb_id': 'PDB_ID',
            'PDB_Code': 'PDB_ID',
            'heteroatom_code': 'Heteroatom_Code',
            'Heteroatom': 'Heteroatom_Code',
            'smiles': 'SMILES',
            'SMILES_String': 'SMILES',
            'chemical_name': 'Chemical_Name',
            'Name': 'Chemical_Name',
            'formula': 'Formula',
            'Molecular_Formula': 'Formula',
            'status': 'Status',
            'chains': 'Chains',
            'residue_numbers': 'Residue_Numbers',
            'atom_count': 'Atom_Count'
        }
        
        # Apply column mappings
        df = df.rename(columns=column_mappings)
        
        # Ensure required columns exist
        required_columns = [
            'PDB_ID', 'Heteroatom_Code', 'SMILES', 'Chemical_Name', 
            'Formula', 'Status', 'Chains', 'Residue_Numbers', 'Atom_Count'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                logger.warning(f"Added missing column: {col}")
        
        return df
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data"""
        
        logger.info("Cleaning and validating data...")
        
        # Filter out entries with no heteroatoms
        original_length = len(df)
        df = df[df['Heteroatom_Code'] != 'NO_HETEROATOMS'].copy()
        filtered_length = len(df)
        
        logger.info(f"Filtered out {original_length - filtered_length} entries with no heteroatoms")
        
        # Clean SMILES strings
        df['SMILES'] = df['SMILES'].fillna('').str.strip()
        df['SMILES'] = df['SMILES'].replace('', None)
        
        # Clean chemical names
        df['Chemical_Name'] = df['Chemical_Name'].fillna('').str.strip()
        df['Chemical_Name'] = df['Chemical_Name'].replace('', 'UNKNOWN')
        
        # Clean formulas
        df['Formula'] = df['Formula'].fillna('').str.strip()
        df['Formula'] = df['Formula'].replace('', 'UNKNOWN')
        
        # Convert atom count to numeric
        df['Atom_Count'] = pd.to_numeric(df['Atom_Count'], errors='coerce').fillna(0).astype(int)
        
        # Clean status
        df['Status'] = df['Status'].fillna('unknown').str.lower()
        
        # Add derived features
        df['has_smiles'] = df['SMILES'].notna() & (df['SMILES'] != '')
        df['molecular_weight_estimated'] = df['Atom_Count'] * 15  # Rough estimate
        
        return df
    
    def _create_search_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create search indexes for faster querying"""
        
        logger.info("Creating search indexes...")
        
        # Create searchable text fields
        df['search_text'] = (
            df['PDB_ID'].fillna('') + ' ' +
            df['Heteroatom_Code'].fillna('') + ' ' +
            df['Chemical_Name'].fillna('') + ' ' +
            df['Formula'].fillna('')
        ).str.lower()
        
        # Create categorical indexes
        df['PDB_ID'] = df['PDB_ID'].astype('category')
        df['Heteroatom_Code'] = df['Heteroatom_Code'].astype('category')
        df['Status'] = df['Status'].astype('category')
        
        return df
    
    def _save_to_sqlite(self, df: pd.DataFrame):
        """Save database to SQLite for faster future loading"""
        
        logger.info(f"Caching database to SQLite: {self.sqlite_db_path}")
        
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                df.to_sql('heteroatoms', conn, if_exists='replace', index=False)
                
                # Create indexes for faster searching
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pdb_id ON heteroatoms(PDB_ID)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_heteroatom_code ON heteroatoms(Heteroatom_Code)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_smiles ON heteroatoms(SMILES)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_has_smiles ON heteroatoms(has_smiles)")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
    
    def _load_from_sqlite(self) -> pd.DataFrame:
        """Load database from SQLite cache"""
        
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM heteroatoms", conn)
                
                # Restore categorical types
                df['PDB_ID'] = df['PDB_ID'].astype('category')
                df['Heteroatom_Code'] = df['Heteroatom_Code'].astype('category')
                df['Status'] = df['Status'].astype('category')
                
                self.database_df = df
                self.stats = self._calculate_statistics(df)
                
                logger.info(f"Loaded {len(df):,} compounds from SQLite cache")
                return df
                
        except Exception as e:
            logger.error(f"Error loading from SQLite: {e}")
            return pd.DataFrame()
    
    def _calculate_statistics(self, df: pd.DataFrame) -> DatabaseStats:
        """Calculate comprehensive database statistics"""
        
        logger.info("Calculating database statistics...")
        
        stats = DatabaseStats(
            total_compounds=len(df),
            total_pdbs=df['PDB_ID'].nunique(),
            compounds_with_smiles=df['has_smiles'].sum(),
            unique_heteroatom_codes=df['Heteroatom_Code'].nunique(),
            average_atom_count=df['Atom_Count'].mean(),
            status_distribution=df['Status'].value_counts().to_dict(),
            formula_distribution=df['Formula'].value_counts().head(20).to_dict()
        )
        
        return stats
    
    def search_compounds(self, 
                        query: str = None,
                        pdb_id: str = None,
                        heteroatom_code: str = None,
                        has_smiles: bool = None,
                        min_atom_count: int = None,
                        max_atom_count: int = None,
                        status: str = None,
                        limit: int = 1000) -> pd.DataFrame:
        """
        Search compounds in the database with various filters
        """
        
        if self.database_df is None:
            self.load_comprehensive_database()
        
        df = self.database_df.copy()
        
        # Apply filters
        if query:
            query_lower = query.lower()
            mask = df['search_text'].str.contains(query_lower, na=False)
            df = df[mask]
        
        if pdb_id:
            df = df[df['PDB_ID'].str.contains(pdb_id, case=False, na=False)]
        
        if heteroatom_code:
            df = df[df['Heteroatom_Code'].str.contains(heteroatom_code, case=False, na=False)]
        
        if has_smiles is not None:
            df = df[df['has_smiles'] == has_smiles]
        
        if min_atom_count is not None:
            df = df[df['Atom_Count'] >= min_atom_count]
        
        if max_atom_count is not None:
            df = df[df['Atom_Count'] <= max_atom_count]
        
        if status:
            df = df[df['Status'].str.contains(status, case=False, na=False)]
        
        # Apply limit
        if limit and len(df) > limit:
            df = df.head(limit)
        
        logger.info(f"Search returned {len(df)} compounds")
        return df
    
    def get_compounds_with_smiles(self, limit: int = None) -> pd.DataFrame:
        """Get all compounds that have valid SMILES strings"""
        
        if self.database_df is None:
            self.load_comprehensive_database()
        
        df = self.database_df[self.database_df['has_smiles'] == True].copy()
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def get_random_compounds(self, n: int = 100, with_smiles: bool = True) -> pd.DataFrame:
        """Get random compounds for testing/sampling"""
        
        if self.database_df is None:
            self.load_comprehensive_database()
        
        df = self.database_df.copy()
        
        if with_smiles:
            df = df[df['has_smiles'] == True]
        
        if len(df) < n:
            logger.warning(f"Requested {n} compounds but only {len(df)} available")
            return df
        
        return df.sample(n=n, random_state=42)
    
    def get_statistics(self) -> DatabaseStats:
        """Get database statistics"""
        
        if self.stats is None and self.database_df is not None:
            self.stats = self._calculate_statistics(self.database_df)
        
        return self.stats
    
    def export_for_ai_analysis(self, 
                              output_file: str = "ai_analysis_dataset.csv",
                              max_compounds: int = 10000,
                              prioritize_smiles: bool = True) -> str:
        """
        Export a curated dataset optimized for AI analysis
        """
        
        if self.database_df is None:
            self.load_comprehensive_database()
        
        df = self.database_df.copy()
        
        # Prioritize compounds with SMILES for AI analysis
        if prioritize_smiles:
            smiles_df = df[df['has_smiles'] == True]
            non_smiles_df = df[df['has_smiles'] == False]
            
            # Take more compounds with SMILES
            smiles_count = min(int(max_compounds * 0.8), len(smiles_df))
            non_smiles_count = min(max_compounds - smiles_count, len(non_smiles_df))
            
            selected_df = pd.concat([
                smiles_df.sample(n=smiles_count, random_state=42) if smiles_count > 0 else pd.DataFrame(),
                non_smiles_df.sample(n=non_smiles_count, random_state=42) if non_smiles_count > 0 else pd.DataFrame()
            ], ignore_index=True)
        else:
            selected_df = df.sample(n=min(max_compounds, len(df)), random_state=42)
        
        # Select key columns for AI analysis
        ai_columns = [
            'PDB_ID', 'Heteroatom_Code', 'SMILES', 'Chemical_Name', 
            'Formula', 'Atom_Count', 'has_smiles', 'Status'
        ]
        
        export_df = selected_df[ai_columns].copy()
        
        # Save to file
        output_path = self.data_folder / output_file
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(export_df)} compounds for AI analysis to {output_path}")
        return str(output_path)

    def get_database_summary(self) -> Dict:
        """Get a comprehensive summary of the database"""
        
        if self.database_df is None:
            self.load_comprehensive_database()
        
        stats = self.get_statistics()
        
        summary = {
            'database_overview': {
                'total_compounds': stats.total_compounds,
                'total_pdb_structures': stats.total_pdbs,
                'compounds_with_smiles': stats.compounds_with_smiles,
                'smiles_coverage_percentage': (stats.compounds_with_smiles / stats.total_compounds) * 100,
                'unique_heteroatom_types': stats.unique_heteroatom_codes,
                'average_atom_count': round(stats.average_atom_count, 2)
            },
            'status_distribution': stats.status_distribution,
            'top_heteroatom_codes': dict(list(self.database_df['Heteroatom_Code'].value_counts().head(10).items())),
            'top_formulas': stats.formula_distribution,
            'atom_count_distribution': {
                'min': int(self.database_df['Atom_Count'].min()),
                'max': int(self.database_df['Atom_Count'].max()),
                'median': int(self.database_df['Atom_Count'].median()),
                'std': round(self.database_df['Atom_Count'].std(), 2)
            }
        }
        
        return summary

# Convenience functions for integration with existing code
def get_comprehensive_database() -> ComprehensiveHeteroatomDatabase:
    """Get a singleton instance of the comprehensive database"""
    if not hasattr(get_comprehensive_database, '_instance'):
        get_comprehensive_database._instance = ComprehensiveHeteroatomDatabase()
    return get_comprehensive_database._instance

def load_heteroatom_data_for_ai(max_compounds: int = 5000) -> pd.DataFrame:
    """
    Load heteroatom data optimized for AI analysis
    This function replaces the mock data in existing AI systems
    """
    db = get_comprehensive_database()
    return db.get_compounds_with_smiles(limit=max_compounds)

def search_heteroatoms(query: str, limit: int = 1000) -> pd.DataFrame:
    """Search heteroatoms - interface for existing code"""
    db = get_comprehensive_database()
    return db.search_compounds(query=query, limit=limit)

def get_random_heteroatoms(n: int = 100) -> pd.DataFrame:
    """Get random heteroatoms for testing - interface for existing code"""
    db = get_comprehensive_database()
    return db.get_random_compounds(n=n, with_smiles=True)

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing Comprehensive Heteroatom Database Loader...")
    
    # Initialize database
    db = ComprehensiveHeteroatomDatabase()
    
    # Load database
    df = db.load_comprehensive_database()
    
    # Print summary
    summary = db.get_database_summary()
    print(json.dumps(summary, indent=2))
    
    # Test search functionality
    print("\nTesting search functionality:")
    
    # Search for ATP
    atp_results = db.search_compounds(query="ATP", limit=10)
    print(f"ATP search results: {len(atp_results)} compounds")
    
    # Search for compounds with SMILES
    smiles_compounds = db.get_compounds_with_smiles(limit=10)
    print(f"Compounds with SMILES: {len(smiles_compounds)} found")
    
    # Export for AI analysis
    ai_file = db.export_for_ai_analysis(max_compounds=1000)
    print(f"Exported AI dataset to: {ai_file}")
    
    logger.info("Database testing completed successfully!")