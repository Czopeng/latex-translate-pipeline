# latex-translation-pipeline

A Python-based pipeline for translating German LaTeX documents into English using LLMs. The system preserves LaTeX structure, integrates a dynamic glossary, and ensures consistent technical terminology through automated glossary extraction and updates.

## Features

- Safe handling of LaTeX commands (e.g. `\gls{}`, `\cite{}`, `\ref{}`)
- Automated glossary parsing and translation
- Consistent terminology enforcement
- Modular pipeline architecture
- Extensible for different LLM providers

## Project Structure
latex-translation-pipeline/
│
├── translator/
│ ├── main.py
│ ├── pipeline.py
│ ├── latex_masker.py
│ ├── glossary.py
│ ├── translator.py
│ └── utils.py
│
├── glossary.json
├── requirements.txt
├── README.md
└── .gitignore



## Setup

```bash
git clone <your-repo-url>
cd latex-translation-pipeline
pip install -r requirements.txt

Create a .env file:
OPENAI_API_KEY=your_api_key_here
```

## Usage
python -m translator.main
