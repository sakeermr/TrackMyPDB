"""
Test Google AI Studio integration
"""
import streamlit as st

def test_gemini_integration():
    """Test function to verify Gemini AI integration"""
    try:
        # Test the config
        from backend.config import Config
        
        st.write("**🔧 Configuration Test:**")
        api_key = Config.get_api_key()
        if api_key:
            st.success(f"✅ API key configured (ends with: ...{api_key[-8:]})")
        else:
            st.error("❌ No API key found")
        
        # Test the Gemini agent
        st.write("**🤖 Gemini Agent Test:**")
        from backend.gemini_agent import GeminiAgent
        
        agent = GeminiAgent()
        if agent.is_available():
            st.success("✅ Gemini agent initialized successfully")
            
            # Test a simple query
            test_query = "Analyze this SMILES: CCO"
            result = agent.process_query_sync(test_query)
            
            st.write("**📊 Test Query Result:**")
            st.json(result)
            
        else:
            st.error("❌ Gemini agent initialization failed")
            
    except Exception as e:
        st.error(f"❌ Integration test failed: {str(e)}")
        st.write("**Error details:**", str(e))

if __name__ == "__main__":
    st.title("🧪 Google AI Studio Integration Test")
    st.write("Testing the integration with your Google AI Studio API key...")
    
    test_gemini_integration()