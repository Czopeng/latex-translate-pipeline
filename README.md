# LaTeX Translate Pipeline

Python project for translating German LaTeX content into English while preserving LaTeX structure and terminology consistency.

## Overview

This repository is prepared as a Python project with:

- dependency management via `requirements.txt`
- Conda environment management via `environment.yml`
- Python-focused `.gitignore`

## Conda Environment Setup

Create and activate the Conda environment named `vEnvTranslate`:

```bash
conda env create -f environment.yml
conda activate vEnvTranslate
```

If you update dependencies later, rebuild with:

```bash
conda env update -f environment.yml --prune
```

This project uses a stable Python version: `3.11`.

## Environment Variables

Create a local `.env` file for API keys and other secrets. This file is ignored by Git.

Example:

```env
OPENAI_API_KEY=your_api_key_here
```

## Suggested Project Layout

```text
latex-translate-pipeline/
|-- environment.yml
|-- .env                    # local secrets (gitignored)
|-- requirements.txt
|-- README.md
`-- .gitignore
```
