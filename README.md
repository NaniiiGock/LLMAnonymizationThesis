
# UV setup + spacy

uv init my_project
cd my_project
uv add spacy
uv run python -m spacy download en_core_web_sm

uv sync

# Ollama setup

install ollama 
>>> ollama run llama3.2

# Run

uv run streamlit run streamlit_app.py