# Anytext-to-Citation

A web API and frontend tool that converts unstructured citation text into structured Hayagriva YAML format and other citation formats (BibTeX, CSL-JSON, RIS) using Large Language Models.

## Overview

Anytext-to-Citation leverages the power of Large Language Models (LLMs) to parse and structure citation data from plain text. Unlike traditional citation parsers that rely on rigid patterns and rules, this tool can handle a wide variety of citation styles and formats by using AI to understand the semantic structure of citations.

### Key Features

- **Flexible Input**: Process citations in virtually any format or style
- **Multiple Output Formats**: Convert to Hayagriva YAML, BibTeX, CSL-JSON, and RIS
- **Web Interface**: User-friendly interface for pasting or uploading citation text
- **API Access**: RESTful API endpoints for integration with other tools
- **Real-time Progress Tracking**: Monitor the conversion process with detailed progress updates

## Similar Tools

This project was inspired by and complements other citation parsing tools:

- [AnyStyle.io](https://anystyle.io): A machine learning-based citation parser that extracts structured references from plain text
- [Text2Bib.org](https://text2bib.org): A tool that converts plain text citations to BibTeX format

Unlike these tools, Anytext-to-Citation uses state-of-the-art LLMs to handle more complex and varied citation formats, and offers multiple output formats including Hayagriva YAML.

## Setup

### Prerequisites

- Python 3.8+
- [Simon Willison's LLM tool](https://github.com/simonw/llm)
- API keys for OpenAI (preferred) or Google Gemini

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/anytext-to-citation.git
   cd anytext-to-citation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .envrc.example .envrc
   # Edit .envrc with your API keys
   ```

   If you use `direnv`, run:
   ```bash
   direnv allow
   ```
   Otherwise, load the environment variables manually.

### Running the Application

```bash
uvicorn convert:app --reload
```

The application will be available at http://localhost:8000

## API Endpoints

### Convert to Hayagriva YAML

```
POST /api/convert-to-hayagriva
```

Request body:
```json
{
  "text": "Your citation text here"
}
```

Response:
```json
{
  "yaml": "# Hayagriva YAML content"
}
```

### Convert Format

```
POST /api/convert-format
```

Request body:
```json
{
  "yaml_content": "Your Hayagriva YAML here",
  "output_format": "bibtex|csl-json|ris"
}
```

Response:
```json
{
  "format": "bibtex|csl-json|ris",
  "converted_content": "Converted content"
}
```

### Check Progress

```
GET /api/progress
```

Response:
```json
{
  "status": "idle|splitting|processing|complete|error",
  "message": "Status message",
  "current": 0,
  "total": 0
}
```

## How It Works

1. The application splits the input text into individual citation entries
2. Each citation is processed by an LLM (OpenAI GPT-4o, GPT-4, GPT-3.5-turbo, or Gemini)
3. The LLM converts the unstructured text into structured Hayagriva YAML format
4. The Hayagriva YAML can then be converted to other formats (BibTeX, CSL-JSON, RIS)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
