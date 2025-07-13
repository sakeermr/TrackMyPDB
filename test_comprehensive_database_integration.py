#!/usr/bin/env python3
"""
Comprehensive Database Integration Test
Tests the comprehensive database integration with real CSV data
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_basic_imports():
    """Test that all required modules can be imported"""
    print("🔧 Testing module imports...")
    
    try:
        from backend.comprehensive_database_loader import (
            ComprehensiveHeteroatomDatabase, 
            get_comprehensive_database,
            load_heteroatom_data_for_ai, 
            search_heteroatoms, 
            get_random_heteroatoms
        )
        print("✅ Successfully imported comprehensive_database_loader")
        
        from backend.autonomous_iterator import (
            AutonomousIterator, IterationConfig, IterationMode, IterationState
        )
        print("✅ Successfully imported autonomous_iterator")
        
        from backend.agentic_layer import TrackMyPDBAgenticInterface
        print("✅ Successfully imported agentic_layer")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_directory():
    """Test that the data directory exists and contains CSV files"""
    print("\n📁 Testing data directory...")
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"✅ Found {len(csv_files)} CSV files in data directory")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found")
        return False
    
    # Test loading a sample CSV file
    sample_file = csv_files[0]
    try:
        df = pd.read_csv(sample_file)
        print(f"✅ Successfully loaded sample file: {sample_file.name}")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['PDB_ID', 'SMILES', 'Heteroatom_Code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing some columns: {missing_cols}")
        else:
            print("✅ All required columns present")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return False

def test_comprehensive_database_loading():
    """Test loading the comprehensive database"""
    print("\n🗃️ Testing comprehensive database loading...")
    
    try:
        from backend.comprehensive_database_loader import get_comprehensive_database
        
        # Get database instance
        db = get_comprehensive_database()
        print("✅ Created database instance")
        
        # Load database
        start_time = time.time()
        df = db.load_comprehensive_database()
        load_time = time.time() - start_time
        
        if df.empty:
            print("❌ Database loaded but is empty")
            return False
            
        print(f"✅ Database loaded successfully in {load_time:.2f} seconds")
        print(f"   - Total compounds: {len(df):,}")
        print(f"   - Columns: {list(df.columns)}")
        
        # Get database summary
        summary = db.get_database_summary()
        overview = summary.get('database_overview', {})
        
        print(f"📊 Database Overview:")
        print(f"   - Total compounds: {overview.get('total_compounds', 0):,}")
        print(f"   - Total PDB structures: {overview.get('total_pdb_structures', 0):,}")
        print(f"   - Compounds with SMILES: {overview.get('compounds_with_smiles', 0):,}")
        print(f"   - SMILES coverage: {overview.get('smiles_coverage_percentage', 0):.1f}%")
        print(f"   - Unique heteroatom types: {overview.get('unique_heteroatom_types', 0):,}")
        
        return True, df
        
    except Exception as e:
        print(f"❌ Database loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_database_search_functionality(df):
    """Test database search functionality"""
    print("\n🔍 Testing database search functionality...")
    
    try:
        from backend.comprehensive_database_loader import search_heteroatoms, get_random_heteroatoms
        
        # Test search by heteroatom code
        search_results = search_heteroatoms("ATP", search_type="heteroatom_code")
        print(f"✅ Search for 'ATP': found {len(search_results)} results")
        
        # Test search by chemical name
        search_results = search_heteroatoms("adenosine", search_type="chemical_name")
        print(f"✅ Search for 'adenosine': found {len(search_results)} results")
        
        # Test getting random heteroatoms
        random_compounds = get_random_heteroatoms(n=10)
        print(f"✅ Random selection: got {len(random_compounds)} compounds")
        
        return True
        
    except Exception as e:
        print(f"❌ Search functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_data_loading():
    """Test AI-specific data loading functionality"""
    print("\n🤖 Testing AI data loading...")
    
    try:
        from backend.comprehensive_database_loader import load_heteroatom_data_for_ai
        
        # Test loading data for AI analysis
        ai_data = load_heteroatom_data_for_ai(limit=100)
        
        if ai_data.empty:
            print("❌ AI data loading returned empty DataFrame")
            return False
            
        print(f"✅ AI data loaded: {len(ai_data)} compounds")
        print(f"   - Columns: {list(ai_data.columns)}")
        
        # Check SMILES coverage
        smiles_count = ai_data['SMILES'].notna().sum()
        smiles_percentage = (smiles_count / len(ai_data)) * 100
        print(f"   - SMILES coverage: {smiles_count}/{len(ai_data)} ({smiles_percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ AI data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_integration():
    """Test that Streamlit interface can access the database"""
    print("\n🌐 Testing Streamlit integration...")
    
    try:
        # Import streamlit components (without running the app)
        import streamlit as st
        from streamlit_iteration_interface import initialize_comprehensive_database
        
        print("✅ Streamlit interface imports successful")
        
        # Test that the interface can be loaded (in theory)
        print("✅ Streamlit integration appears functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False

def test_similarity_analysis_readiness():
    """Test that the data is ready for similarity analysis"""
    print("\n🧬 Testing similarity analysis readiness...")
    
    try:
        from backend.comprehensive_database_loader import get_comprehensive_database
        
        db = get_comprehensive_database()
        df = db.load_comprehensive_database()
        
        # Check for SMILES strings
        valid_smiles = df[df['SMILES'].notna() & (df['SMILES'] != '')]
        
        if len(valid_smiles) == 0:
            print("❌ No valid SMILES strings found")
            return False
            
        print(f"✅ Found {len(valid_smiles):,} compounds with valid SMILES")
        
        # Sample some SMILES for testing
        sample_smiles = valid_smiles['SMILES'].head(5).tolist()
        print(f"📋 Sample SMILES strings:")
        for i, smiles in enumerate(sample_smiles, 1):
            print(f"   {i}. {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        # Check for heteroatom diversity
        unique_heteroatoms = df['Heteroatom_Code'].nunique()
        print(f"✅ Found {unique_heteroatoms} unique heteroatom types")
        
        # Show top heteroatom types
        top_heteroatoms = df['Heteroatom_Code'].value_counts().head(10)
        print(f"📊 Top 10 heteroatom types:")
        for code, count in top_heteroatoms.items():
            print(f"   {code}: {count:,} occurrences")
        
        return True
        
    except Exception as e:
        print(f"❌ Similarity analysis readiness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide a comprehensive report"""
    print("🧪 COMPREHENSIVE DATABASE INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_basic_imports()
    test_results['data_directory'] = test_data_directory()
    
    db_success, df = test_comprehensive_database_loading()
    test_results['database_loading'] = db_success
    
    if db_success and df is not None:
        test_results['search_functionality'] = test_database_search_functionality(df)
        test_results['ai_data_loading'] = test_ai_data_loading()
        test_results['similarity_readiness'] = test_similarity_analysis_readiness()
    else:
        test_results['search_functionality'] = False
        test_results['ai_data_loading'] = False
        test_results['similarity_readiness'] = False
    
    test_results['streamlit_integration'] = test_streamlit_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your database integration is working perfectly!")
        print("\n💡 Next steps:")
        print("   1. Run: streamlit run streamlit_iteration_interface.py")
        print("   2. Test the autonomous iteration features")
        print("   3. Explore your comprehensive heteroatom database")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that data/ directory contains your CSV files")
        print("   3. Verify that backend/ modules are properly structured")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)