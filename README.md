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
geminiAPIkey=your_api_key_here
```

## Workflow Summary

The pipeline supports single-file and folder-scan translation workflows.

Core behavior:

- glossary source LaTeX (`--glossary-source`) is treated as read-only ground truth
- glossary source is parsed and synced into `glossary.json` while preserving approved English values
- content is chunked by `\chapter`, `\section`, and `\subsection`
- oversized chunks are split further with paragraph-aware fallback
- each chunk is translated with placeholder integrity checks and strict LaTeX restoration

## First-Run Rollout

Use this sequence for the first end-to-end test.

1. Check Gemini connectivity before long runs:

```bash
python -m translator.main \
	--provider gemini \
	--test-gemini-connection
```

2. Optional dry-run (manifest and queues only, no translated file writes):

```bash
python -m translator.main \
	--input-folder testLatex \
	--output-folder translated_out \
	--glossary-source testLatex/glossarentries.tex \
	--provider gemini \
	--dry-run \
	--manifest-output run_manifest_dry.json
```

3. Full rollout run (writes translated LaTeX and translated glossary artifacts):

```bash
python -m translator.main \
	--input-folder testLatex \
	--output-folder translated_out \
	--glossary-source testLatex/glossarentries.tex \
	--provider gemini \
	--manifest-output run_manifest.json
```

Full rollout output folder includes:

- translated document tree (mirrored `.tex` files)
- `translated_glossarentries.tex`
- `translated_glossary.json`

## Folder Translation Mode

Translate all `.tex` files under a folder (excluding the glossary source file itself):

```bash
python -m translator.main \
	--input-folder testLatex \
	--output-folder translated_out \
	--glossary-source testLatex/glossarentries.tex \
	--provider gemini \
	--manifest-output run_manifest.json
```

Optional term-review queue generation from translated outputs:

```bash
python -m translator.main \
	--input-folder testLatex \
	--output-folder translated_out \
	--glossary-source testLatex/glossarentries.tex \
	--provider gemini \
	--build-term-review-queue \
	--term-review-output term_review_queue.json \
	--term-min-frequency 2
```

Dry-run folder planning (no translated file writes, but manifest/queues are generated):

```bash
python -m translator.main \
	--input-folder testLatex \
	--output-folder translated_out \
	--glossary-source testLatex/glossarentries.tex \
	--provider gemini \
	--dry-run \
	--build-term-review-queue \
	--term-review-output term_review_queue.json \
	--stale-report-output stale_glossary_report.json \
	--manifest-output run_manifest.json
```

## Glossary Review Lifecycle

1. Generate glossary translation suggestions queue:

```bash
python -m translator.main testLatex/glossarentries.tex \
	--provider gemini \
	--build-glossary-review-queue \
	--review-output glossary_review_queue.json
```

2. Manually approve items in `glossary_review_queue.json` by setting `status` to `approved`.

3. Apply approved items only:

```bash
python -m translator.main testLatex/glossarentries.tex \
	--apply-glossary-review-queue \
	--review-input glossary_review_queue.json \
	--glossary glossary.json
```

Note: New candidate terms are generated into a term review queue and are not auto-committed to `glossary.json`.

## Output Conventions

- Folder mode writes translated files to a mirrored path under `--output-folder`
- Optional manifest (`--manifest-output`) includes:
	- processed file list
	- dry-run marker and per-file write status
	- per-file chunk count and duration
	- glossary sync and glossary conflict summary
	- stale glossary summary (non-source terms) and optional stale report path
	- glossary delta counters
	- optional term-review queue summary

## Suggested Project Layout

```text
latex-translate-pipeline/
|-- environment.yml
|-- .env                    # local secrets (gitignored)
|-- requirements.txt
|-- README.md
`-- .gitignore
```
