"""
Test Script for Comprehensive Database Integration
Tests the integration of the comprehensive heteroatom database with real CSV data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import asyncio
import time
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from backend.comprehensive_database_loader import (
        ComprehensiveHeteroatomDatabase, 
        get_comprehensive_database,
        load_heteroatom_data_for_ai,
        search_heteroatoms,
        get_random_heteroatoms
    )
    DATABASE_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing database module: {e}")
    DATABASE_MODULE_AVAILABLE = False

def test_database_discovery():
    """Test CSV file discovery in the data directory"""
    print("🔍 Testing CSV file discovery...")
    
    data_folder = Path("data")
    if not data_folder.exists():
        print(f"❌ Data folder not found: {data_folder}")
        return False
    
    csv_files = list(data_folder.glob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files in data directory")
    
    # Show first 10 files
    for i, file in enumerate(csv_files[:10]):
        print(f"  {i+1}. {file.name}")
    
    if len(csv_files) > 10:
        print(f"  ... and {len(csv_files) - 10} more files")
    
    return len(csv_files) > 0

def test_single_csv_loading():
    """Test loading a single CSV file"""
    print("\n📄 Testing single CSV file loading...")
    
    data_folder = Path("data")
    csv_files = list(data_folder.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found to test")
        return False
    
    # Test loading the first CSV file
    test_file = csv_files[0]
    print(f"📂 Testing file: {test_file.name}")
    
    try:
        df = pd.read_csv(test_file)
        print(f"✅ Successfully loaded {test_file.name}")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Column names: {list(df.columns)}")
        
        # Check for key columns
        key_columns = ['PDB_ID', 'SMILES', 'Chemical_Name', 'Heteroatom_Code']
        available_columns = [col for col in key_columns if col in df.columns]
        print(f"   - Key columns available: {available_columns}")
        
        # Show sample data
        if not df.empty:
            print(f"   - Sample row:")
            for col in df.columns[:5]:  # Show first 5 columns
                value = df.iloc[0][col] if len(df) > 0 else "N/A"
                print(f"     {col}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {test_file.name}: {e}")
        return False

def test_database_initialization():
    """Test comprehensive database initialization"""
    print("\n🗃️ Testing comprehensive database initialization...")
    
    if not DATABASE_MODULE_AVAILABLE:
        print("❌ Database module not available")
        return False
    
    try:
        # Initialize database
        db = ComprehensiveHeteroatomDatabase()
        print("✅ Database object created successfully")
        
        # Test file discovery
        csv_files = db.discover_csv_files()
        print(f"✅ Discovered {len(csv_files)} CSV files")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_database_loading():
    """Test loading the complete database"""
    print("\n📊 Testing comprehensive database loading...")
    
    if not DATABASE_MODULE_AVAILABLE:
        print("❌ Database module not available")
        return False
    
    try:
        start_time = time.time()
        
        # Get database instance
        db = get_comprehensive_database()
        print("✅ Got database singleton instance")
        
        # Load database
        print("🔄 Loading comprehensive database...")
        df = db.load_comprehensive_database()
        
        load_time = time.time() - start_time
        
        if df.empty:
            print("❌ Database loaded but is empty")
            return False
        
        print(f"✅ Database loaded successfully in {load_time:.2f} seconds")
        print(f"   - Total compounds: {len(df):,}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Test data quality
        smiles_count = df['SMILES'].notna().sum() if 'SMILES' in df.columns else 0
        print(f"   - Compounds with SMILES: {smiles_count:,}")
        
        if 'Heteroatom_Code' in df.columns:
            unique_codes = df['Heteroatom_Code'].nunique()
            print(f"   - Unique heteroatom codes: {unique_codes:,}")
            
            # Show top heteroatom codes
            top_codes = df['Heteroatom_Code'].value_counts().head(5)
            print(f"   - Top 5 heteroatom codes:")
            for code, count in top_codes.items():
                print(f"     {code}: {count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_statistics():
    """Test database statistics generation"""
    print("\n📈 Testing database statistics...")
    
    if not DATABASE_MODULE_AVAILABLE:
        print("❌ Database module not available")
        return False
    
    try:
        db = get_comprehensive_database()
        
        # Load database if not already loaded
        if db.database_df is None:
            df = db.load_comprehensive_database()
            if df.empty:
                print("❌ Database is empty")
                return False
        
        # Get statistics
        stats = db.get_statistics()
        if stats is None:
            print("❌ Statistics generation failed")
            return False
        
        print("✅ Database statistics generated successfully")
        print(f"   - Total compounds: {stats.total_compounds:,}")
        print(f"   - Total PDB structures: {stats.total_pdbs:,}")
        print(f"   - Compounds with SMILES: {stats.compounds_with_smiles:,}")
        print(f"   - Unique heteroatom codes: {stats.unique_heteroatom_codes:,}")
        print(f"   - Average atom count: {stats.average_atom_count:.2f}")
        
        # Get comprehensive summary
        summary = db.get_database_summary()
        print("\n📋 Database Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_search():
    """Test database search functionality"""
    print("\n🔍 Testing database search functionality...")
    
    if not DATABASE_MODULE_AVAILABLE:
        print("❌ Database module not available")
        return False
    
    try:
        db = get_comprehensive_database()
        
        # Ensure database is loaded
        if db.database_df is None:
            df = db.load_comprehensive_database()
            if df.empty:
                print("❌ Database is empty")
                return False
        
        # Test search for ATP
        print("🔍 Searching for 'ATP'...")
        atp_results = db.search_compounds(query="ATP", limit=10)
        print(f"✅ Found {len(atp_results)} results for 'ATP'")
        
        if not atp_results.empty:
            print("   Sample results:")
            for idx, row in atp_results.head(3).iterrows():
                pdb_id = row.get('PDB_ID', 'Unknown')
                het_code = row.get('Heteroatom_Code', 'Unknown')
                chem_name = row.get('Chemical_Name', 'Unknown')
                print(f"     {pdb_id} - {het_code} - {chem_name}")
        
        # Test search with SMILES filter
        print("\n🔍 Searching for compounds with SMILES...")
        smiles_results = db.get_compounds_with_smiles(limit=5)
        print(f"✅ Found {len(smiles_results)} compounds with SMILES")
        
        if not smiles_results.empty:
            print("   Sample SMILES:")
            for idx, row in smiles_results.head(3).iterrows():
                smiles = row.get('SMILES', 'Unknown')
                print(f"     {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        # Test random compound selection
        print("\n🎲 Getting random compounds...")
        random_results = db.get_random_compounds(n=5, with_smiles=True)
        print(f"✅ Got {len(random_results)} random compounds")
        
        return True
        
    except Exception as e:
        print(f"❌ Database search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_integration_functions():
    """Test the AI integration helper functions"""
    print("\n🤖 Testing AI integration functions...")
    
    if not DATABASE_MODULE_AVAILABLE:
        print("❌ Database module not available")
        return False
    
    try:
        # Test load_heteroatom_data_for_ai function
        print("🔄 Testing load_heteroatom_data_for_ai...")
        ai_data = load_heteroatom_data_for_ai(max_compounds=100)
        
        if ai_data.empty:
            print("❌ AI data loading returned empty dataset")
            return False
        
        print(f"✅ AI data loaded: {len(ai_data):,} compounds")
        print(f"   - Columns: {list(ai_data.columns)}")
        
        # Test search_heteroatoms function
        print("\n🔍 Testing search_heteroatoms...")
        search_results = search_heteroatoms(query="kinase", limit=10)
        print(f"✅ Search completed: {len(search_results)} results")
        
        # Test get_random_heteroatoms function
        print("\n🎲 Testing get_random_heteroatoms...")
        random_compounds = get_random_heteroatoms(n=5)
        print(f"✅ Random selection: {len(random_compounds)} compounds")
        
        return True
        
    except Exception as e:
        print(f"❌ AI integration functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agentic_integration():
    """Test integration with agentic layer"""
    print("\n🤖 Testing agentic layer integration...")
    
    try:
        from backend.agentic_layer import TrackMyPDBAgenticInterface
        
        # Initialize agentic interface
        interface = TrackMyPDBAgenticInterface()
        print("✅ Agentic interface initialized")
        
        # Check database status
        db_status = interface.get_database_status()
        print("✅ Database status retrieved:")
        print(json.dumps(db_status, indent=2, default=str))
        
        return True
        
    except ImportError as e:
        print(f"❌ Agentic layer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Agentic integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_csv_files():
    """Test loading the main pipeline CSV files"""
    print("\n🚀 Testing main pipeline CSV files...")
    
    pipeline_files = [
        "complete_pipeline_heteroatoms.csv",
        "complete_pipeline_similarity.csv"
    ]
    
    results = {}
    
    for filename in pipeline_files:
        file_path = Path(filename)
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                results[filename] = {
                    'status': 'success',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns)
                }
                print(f"✅ {filename}: {len(df):,} rows, {len(df.columns)} columns")
                
                # Show column details
                if not df.empty:
                    for col in df.columns[:5]:  # First 5 columns
                        non_null = df[col].notna().sum()
                        print(f"     {col}: {non_null:,}/{len(df):,} values")
                
            except Exception as e:
                results[filename] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"❌ {filename}: {e}")
        else:
            results[filename] = {
                'status': 'not_found'
            }
            print(f"⚠️ {filename}: File not found")
    
    return results

def run_comprehensive_tests():
    """Run all comprehensive database tests"""
    print("🧪 Starting Comprehensive Database Integration Tests")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: CSV file discovery
    test_results['file_discovery'] = test_database_discovery()
    
    # Test 2: Single CSV loading
    test_results['single_csv_loading'] = test_single_csv_loading()
    
    # Test 3: Database initialization
    test_results['database_initialization'] = test_database_initialization()
    
    # Test 4: Database loading
    test_results['database_loading'] = test_database_loading()
    
    # Test 5: Database statistics
    test_results['database_statistics'] = test_database_statistics()
    
    # Test 6: Database search
    test_results['database_search'] = test_database_search()
    
    # Test 7: AI integration functions
    test_results['ai_integration'] = test_ai_integration_functions()
    
    # Test 8: Pipeline CSV files
    test_results['pipeline_files'] = test_pipeline_csv_files()
    
    # Test 9: Agentic integration (async)
    try:
        test_results['agentic_integration'] = asyncio.run(test_agentic_integration())
    except Exception as e:
        print(f"❌ Agentic integration test failed: {e}")
        test_results['agentic_integration'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result is True else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Database integration is working correctly.")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️ Most tests passed. Some minor issues may need attention.")
    else:
        print("❌ Multiple test failures. Database integration needs fixing.")
    
    return test_results

if __name__ == "__main__":
    # Run all tests
    results = run_comprehensive_tests()
    
    # Additional diagnostics
    print("\n🔧 DIAGNOSTIC INFORMATION")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")
    
    # Check required dependencies
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas: Not installed")
    
    try:
        import numpy
        print(f"✅ Numpy: {numpy.__version__}")
    except ImportError:
        print("❌ Numpy: Not installed")
    
    # Check data directory structure
    data_dir = Path("data")
    if data_dir.exists():
        csv_count = len(list(data_dir.glob("*.csv")))
        print(f"✅ Data directory: {csv_count} CSV files")
    else:
        print("❌ Data directory: Not found")
    
    print("\n🔍 For more details, check the test output above.")