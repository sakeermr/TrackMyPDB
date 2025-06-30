# 🚀 TrackMyPDB - Quick Startup Guide

## Step 1: Test Installation
```bash
python test_installation.py
```

## Step 2: Install Missing Dependencies (if needed)
```bash
# Install basic dependencies
pip install streamlit requests pandas tqdm numpy matplotlib seaborn plotly

# Install RDKit (choose one method)
pip install rdkit-pypi
# OR
conda install -c conda-forge rdkit
```

## Step 3: Launch Application
```bash
streamlit run streamlit_app.py
```

## Step 4: Access the Application
- Open your browser to: `http://localhost:8501`
- Or use the provided batch files:
  - Double-click `install.bat` to install
  - Double-click `run.bat` to launch

## 🧪 Quick Test Data

### UniProt IDs (copy-paste ready):
```
Q9UNQ0, P37231, P06276
```

### Sample SMILES:
```
CCO
```

## 📊 What to Expect
1. **Heteroatom Extraction**: 5-15 minutes for 3 proteins
2. **Similarity Analysis**: 1-2 minutes after extraction
3. **CSV Downloads**: Available after each step

## 🆘 Troubleshooting
- **RDKit Issues**: Use `conda install -c conda-forge rdkit`
- **Slow Performance**: Start with fewer UniProt IDs
- **API Timeouts**: Wait and retry

**Happy analyzing!** 🧬 