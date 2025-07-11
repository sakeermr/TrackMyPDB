# TrackMyPDB

A Python-based tool for analyzing protein-ligand interactions and performing heteroatom analysis using PDB data.

## Features

- Natural language interface for protein analysis
- Heteroatom extraction and analysis
- Similarity analysis for protein-ligand interactions
- Interactive visualizations using Plotly
- Streamlit-based user interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the application:

```bash
streamlit run streamlit_app.py
```

## Testing

Run tests using pytest:

```bash
pytest
```

## Project Structure

```
├── backend/
│   ├── __init__.py
│   ├── agent_core.py
│   ├── heteroatom_extractor.py
│   ├── nl_interface.py
│   ├── similarity_analyzer_simple.py
│   └── similarity_analyzer.py
├── tests/
│   ├── __init__.py
│   └── test_*.py
├── requirements.txt
└── streamlit_app.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.