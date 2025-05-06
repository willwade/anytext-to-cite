from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from pydantic import BaseModel, Field
from llm import get_model
import yaml  # type: ignore
import re
import uuid
import time
from functools import wraps
from pathlib import Path
from enum import Enum

# Import other modules as needed in specific functions
# bibtexparser is imported in the specific functions where it's needed


# Define the request models
class TextInput(BaseModel):
    text: str


class FormatConversionInput(BaseModel):
    yaml_content: str
    output_format: str = Field(
        ..., description="Target format: 'bibtex', 'csl-json', or 'ris'"
    )


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    content: str


# Define output format enum
class OutputFormat(str, Enum):
    YAML = "yaml"
    BIBTEX = "bibtex"
    CSL_JSON = "csl-json"
    RIS = "ris"


# Initialize FastAPI
app = FastAPI(
    title="Hayagriva Converter API",
    description="Convert unstructured citation text to structured Hayagriva YAML and other formats",
    version="0.1.0",
)

# Global progress tracking
conversion_progress = {"total": 0, "current": 0, "status": "idle", "message": ""}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load the LLM model
# Try different models in order of preference, prioritizing Gemini Flash (free tier)
model = None
# List of models to try, in order of preference
model_options = [
    "gemma-3-27b-it",  # Google's Gemini Flash (free free tier)"gemini-2.0-flash-lite",  # Google's Gemini Flash (free free tier)
    "gemini-2.0-flash",  # Google's Gemini Pro
    "gpt-3.5-turbo",  # OpenAI's GPT-3.5 (cheaper than GPT-4)
    "gpt-4o",  # OpenAI's latest model
    "gpt-4",  # OpenAI's GPT-4
    "llama3",  # Meta's Llama 3
]

for model_name in model_options:
    try:
        print(f"Attempting to load model: {model_name}")
        model = get_model(model_name)
        print(f"Successfully loaded {model_name} model")
        break
    except Exception as e:
        print(f"Warning: Could not load {model_name} model: {e}")

# If all models fail, fall back to default model
if model is None:
    print("Falling back to default model")
    try:
        model = get_model()  # Fallback to default model
        print("Successfully loaded default model")
    except Exception as e:
        print(f"Error loading default model: {e}")
        # Set a flag to show an error message in the UI
        model_load_error = True

# Rate limiting configuration
RATE_LIMIT = {
    "requests_per_minute": 60,  # Maximum requests per minute
    "request_timestamps": [],  # Timestamps of recent requests
}


def rate_limited_api_call(func):
    """Decorator to apply rate limiting to API calls"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clean up old timestamps (older than 1 minute)
        current_time = time.time()
        RATE_LIMIT["request_timestamps"] = [
            ts for ts in RATE_LIMIT["request_timestamps"] if current_time - ts < 60
        ]

        # Check if we've exceeded the rate limit
        if len(RATE_LIMIT["request_timestamps"]) >= RATE_LIMIT["requests_per_minute"]:
            # Calculate time to wait
            oldest_timestamp = min(RATE_LIMIT["request_timestamps"])
            wait_time = 60 - (current_time - oldest_timestamp)

            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)

        # Add current timestamp to the list
        RATE_LIMIT["request_timestamps"].append(time.time())

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


# Improved and robust function to split text into reference entries
def split_references(text):
    # Patterns for different reference formats

    # 1. Numbered references (e.g., "103. Title...")
    numbered_pattern = re.compile(r"(?:^|\n)(\d+\.\s.+?)(?=\n\d+\.|\Z)", re.DOTALL)

    # 2. Unnumbered references with quotes in title
    unnumbered_pattern = re.compile(
        r"(?:^|\n)([^\n]+?\.\s+\".+?\".+?)(?=\n[^\n]+?\.\s\"|\Z)", re.DOTALL
    )

    # 3. References with "accessed on" and URL
    simple_url_pattern = re.compile(
        r"(?:^|\n)(.+?,\saccessed on.+?https?://[^\s]+)(?=\n|$)", re.DOTALL
    )

    # 4. References with URL at the end (without "accessed on")
    # More specific to avoid matching too much text
    url_ending_pattern = re.compile(
        r"(?:^|\n)([^.\n]+\.[^.\n]+(?:https?://[^\s]+))(?=\n|$)", re.DOTALL
    )

    # 5. Author-date format with DOI/URL at end
    author_date_pattern = re.compile(
        r"(?:^|\n)([A-Za-z\s,\.]+?\(\d{4}\)\..+?(?:doi|https?://).+?)"
        r"(?=\n[A-Za-z\s,\.]+?\(\d{4}\)\.|\Z)",
        re.DOTALL,
    )

    # 6. Academic citation format (Author, Year, Title, Journal, etc.)
    academic_pattern = re.compile(
        r"(?:^|\n)([A-Za-z][^,\n]*,[^,\n]*(?:et al\.)?[^,\n]*\(\d{4}\)[^.\n]*\."
        r"[^.\n]*\.[^.\n]*(?:https?://|doi:)[^\s]+)"
        r"(?=\n\s*\n|\n[A-Za-z][^,\n]*,[^,\n]*(?:et al\.)?[^,\n]*\(\d{4}\)|\Z)",
        re.DOTALL,
    )

    entries = []

    # Try each pattern in order of specificity

    # First check if this looks like academic citations (author-year format with blank lines)
    # This is a heuristic check before trying regex patterns
    academic_citation_pattern = r"[A-Za-z][^,\n]*,[^,\n]*(?:et al\.)?[^,\n]*\(\d{4}\)"
    if re.search(academic_citation_pattern, text):
        # If it contains author-year patterns, try splitting by blank lines first
        blank_line_entries = re.split(r"\n\s*\n", text.strip())
        if len(blank_line_entries) > 1:
            # Verify each entry looks like a citation (contains year in parentheses)
            valid_entries = []
            for entry in blank_line_entries:
                entry = entry.strip()
                if entry and re.search(r"\(\d{4}\)", entry):
                    valid_entries.append(entry)

            if len(valid_entries) > 0:
                return valid_entries

    # First try numbered references
    entries = numbered_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # Then try academic citation format
    entries = academic_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # Then try author-date format
    entries = author_date_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # Then try unnumbered (author-title) references
    entries = unnumbered_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # Then try references with "accessed on" and URL
    entries = simple_url_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # Then try references ending with URL
    entries = url_ending_pattern.findall(text)
    if entries:
        return [entry.strip() for entry in entries]

    # If none matched, try a more aggressive approach - split by blank lines
    blank_line_entries = re.split(r"\n\s*\n", text.strip())
    if len(blank_line_entries) > 1:
        return [entry.strip() for entry in blank_line_entries if entry.strip()]

    # Last resort: split by lines
    return [line.strip() for line in text.strip().split("\n") if line.strip()]


def hayagriva_to_bibtex(hayagriva_yaml):
    """Convert Hayagriva YAML to BibTeX format"""
    try:
        # Clean up the YAML before parsing
        cleaned_yaml = hayagriva_yaml.strip()

        # Parse the YAML
        try:
            data = yaml.safe_load(cleaned_yaml)
            if data is None:
                raise ValueError("Empty or invalid YAML content")
            if not isinstance(data, dict):
                raise ValueError(f"Expected a dictionary, got {type(data).__name__}")
        except Exception as yaml_error:
            print(f"YAML parsing error: {str(yaml_error)}")
            print(f"Problematic YAML: {cleaned_yaml[:100]}...")
            raise ValueError(f"Invalid YAML format: {str(yaml_error)}")

        bibtex_entries = []

        for key, entry in data.items():
            entry_type = entry.get("type", "misc").lower()

            # Map Hayagriva types to BibTeX types
            type_mapping = {
                "article": "article",
                "book": "book",
                "chapter": "inbook",
                "thesis": "phdthesis",
                "report": "techreport",
                "conference": "conference",
                "proceedings": "proceedings",
                "anthology": "collection",
                "web": "misc",
                "video": "misc",
                "audio": "misc",
                "patent": "patent",
                "misc": "misc",
            }

            bibtex_type = type_mapping.get(entry_type, "misc")

            # Handle the date/year field properly
            year = ""
            if "date" in entry:
                date_value = entry.get("date")
                if isinstance(date_value, str) and date_value:
                    # If it's a string like "2023-01-01", extract the year
                    year = date_value.split("-")[0]
                elif isinstance(date_value, int):
                    # If it's just a year as an integer
                    year = str(date_value)
                else:
                    # Handle any other case
                    year = str(date_value) if date_value is not None else ""

            # Create BibTeX entry
            bibtex_entry = {
                "ID": key,
                "ENTRYTYPE": bibtex_type,
                "title": entry.get("title", ""),
                "year": year,
            }

            # Handle author field
            if "author" in entry:
                if isinstance(entry["author"], list):
                    bibtex_entry["author"] = " and ".join(entry["author"])
                else:
                    bibtex_entry["author"] = entry["author"]

            # Handle other common fields
            if "publisher" in entry:
                if (
                    isinstance(entry["publisher"], dict)
                    and "name" in entry["publisher"]
                ):
                    bibtex_entry["publisher"] = entry["publisher"]["name"]
                else:
                    bibtex_entry["publisher"] = entry["publisher"]

            if "volume" in entry:
                bibtex_entry["volume"] = str(entry["volume"])

            if "issue" in entry:
                bibtex_entry["number"] = str(entry["issue"])

            if "page-range" in entry:
                bibtex_entry["pages"] = entry["page-range"]

            if "url" in entry:
                if isinstance(entry["url"], dict) and "value" in entry["url"]:
                    bibtex_entry["url"] = entry["url"]["value"]
                else:
                    bibtex_entry["url"] = entry["url"]

            if "serial-number" in entry:
                if isinstance(entry["serial-number"], dict):
                    if "doi" in entry["serial-number"]:
                        bibtex_entry["doi"] = entry["serial-number"]["doi"]
                    if "isbn" in entry["serial-number"]:
                        bibtex_entry["isbn"] = entry["serial-number"]["isbn"]
                    if "issn" in entry["serial-number"]:
                        bibtex_entry["issn"] = entry["serial-number"]["issn"]
                else:
                    bibtex_entry["note"] = f"Serial: {entry['serial-number']}"

            # Handle parent information
            if "parent" in entry:
                parent = entry["parent"]
                if isinstance(parent, dict):
                    if "title" in parent:
                        if bibtex_type == "article":
                            bibtex_entry["journal"] = parent["title"]
                        elif bibtex_type in ["inbook", "incollection"]:
                            bibtex_entry["booktitle"] = parent["title"]

                # Handle multiple parents
                elif isinstance(parent, list):
                    for p in parent:
                        if isinstance(p, dict) and "title" in p:
                            if p.get("type", "").lower() == "periodical":
                                bibtex_entry["journal"] = p["title"]
                            elif p.get("type", "").lower() in ["book", "anthology"]:
                                bibtex_entry["booktitle"] = p["title"]

            bibtex_entries.append(bibtex_entry)

        # Convert to BibTeX string manually
        bibtex_output = []

        for entry in bibtex_entries:
            # Extract ID and entry type
            entry_id = entry.pop("ID", f"ref_{len(bibtex_output) + 1}")
            entry_type = entry.pop("ENTRYTYPE", "misc")

            # Start BibTeX entry
            bibtex_entry = [f"@{entry_type}{{{entry_id},"]

            # Add fields
            for field, value in entry.items():
                if value:  # Only add non-empty fields
                    # Escape special characters in the value
                    value_str = str(value).replace('"', '\\"')
                    bibtex_entry.append(f"  {field} = {{{value_str}}},")

            # Close the entry
            bibtex_entry.append("}")

            # Add to output
            bibtex_output.append("\n".join(bibtex_entry))

        # Join all entries with newlines
        return "\n\n".join(bibtex_output)

    except Exception as e:
        raise ValueError(f"Error converting to BibTeX: {str(e)}")


def hayagriva_to_csl_json(hayagriva_yaml):
    """Convert Hayagriva YAML to CSL-JSON format"""
    try:
        # Clean up the YAML before parsing
        cleaned_yaml = hayagriva_yaml.strip()

        # Parse the YAML
        try:
            data = yaml.safe_load(cleaned_yaml)
            if data is None:
                raise ValueError("Empty or invalid YAML content")
            if not isinstance(data, dict):
                raise ValueError(f"Expected a dictionary, got {type(data).__name__}")
        except Exception as yaml_error:
            print(f"YAML parsing error: {str(yaml_error)}")
            print(f"Problematic YAML: {cleaned_yaml[:100]}...")
            raise ValueError(f"Invalid YAML format: {str(yaml_error)}")

        csl_entries = []

        for key, entry in data.items():
            entry_type = entry.get("type", "misc").lower()

            # Map Hayagriva types to CSL types
            type_mapping = {
                "article": "article-journal",
                "book": "book",
                "chapter": "chapter",
                "thesis": "thesis",
                "report": "report",
                "conference": "paper-conference",
                "proceedings": "paper-conference",
                "anthology": "book",
                "web": "webpage",
                "video": "motion_picture",
                "audio": "song",
                "patent": "patent",
                "misc": "article",
            }

            csl_type = type_mapping.get(entry_type, "article")

            # Create CSL-JSON entry
            csl_entry = {
                "id": key,
                "type": csl_type,
                "title": entry.get("title", ""),
            }

            # Handle date
            if "date" in entry:
                date_value = entry["date"]
                date_parts = []

                if isinstance(date_value, str) and date_value:
                    # If it's a string like "2023-01-01", parse it
                    date_components = date_value.split("-")
                    if len(date_components) >= 1:
                        try:
                            date_parts.append(int(date_components[0]))
                            if len(date_components) >= 2:
                                date_parts.append(int(date_components[1]))
                                if len(date_components) >= 3:
                                    date_parts.append(int(date_components[2]))
                        except ValueError:
                            # If we can't convert to int, use the string as is
                            date_parts = [date_value]
                elif isinstance(date_value, int):
                    # If it's just a year as an integer
                    date_parts = [date_value]
                elif date_value is not None:
                    # Handle any other case
                    date_parts = [str(date_value)]

                if date_parts:
                    csl_entry["issued"] = {"date-parts": [date_parts]}

            # Handle author field
            if "author" in entry:
                authors = []
                if isinstance(entry["author"], list):
                    for author in entry["author"]:
                        if isinstance(author, str):
                            parts = author.split(",", 1)
                            if len(parts) > 1:
                                authors.append(
                                    {
                                        "family": parts[0].strip(),
                                        "given": parts[1].strip(),
                                    }
                                )
                            else:
                                authors.append({"family": author.strip()})
                        elif isinstance(author, dict):
                            author_obj = {}
                            if "name" in author:
                                author_obj["family"] = author["name"]
                            if "given-name" in author:
                                author_obj["given"] = author["given-name"]
                            authors.append(author_obj)
                else:
                    if isinstance(entry["author"], str):
                        parts = entry["author"].split(",", 1)
                        if len(parts) > 1:
                            authors.append(
                                {"family": parts[0].strip(), "given": parts[1].strip()}
                            )
                        else:
                            authors.append({"family": entry["author"].strip()})

                csl_entry["author"] = authors

            # Handle other common fields
            if "publisher" in entry:
                if (
                    isinstance(entry["publisher"], dict)
                    and "name" in entry["publisher"]
                ):
                    csl_entry["publisher"] = entry["publisher"]["name"]
                    if "location" in entry["publisher"]:
                        csl_entry["publisher-place"] = entry["publisher"]["location"]
                else:
                    csl_entry["publisher"] = entry["publisher"]

            if "volume" in entry:
                csl_entry["volume"] = str(entry["volume"])

            if "issue" in entry:
                csl_entry["issue"] = str(entry["issue"])

            if "page-range" in entry:
                csl_entry["page"] = entry["page-range"]

            if "url" in entry:
                if isinstance(entry["url"], dict) and "value" in entry["url"]:
                    csl_entry["URL"] = entry["url"]["value"]
                    if "date" in entry["url"]:
                        csl_entry["accessed"] = {
                            "date-parts": [
                                [int(x) for x in entry["url"]["date"].split("-")]
                            ]
                        }
                else:
                    csl_entry["URL"] = entry["url"]

            if "serial-number" in entry:
                if isinstance(entry["serial-number"], dict):
                    if "doi" in entry["serial-number"]:
                        csl_entry["DOI"] = entry["serial-number"]["doi"]
                    if "isbn" in entry["serial-number"]:
                        csl_entry["ISBN"] = entry["serial-number"]["isbn"]
                    if "issn" in entry["serial-number"]:
                        csl_entry["ISSN"] = entry["serial-number"]["issn"]

            # Handle parent information
            if "parent" in entry:
                parent = entry["parent"]
                if isinstance(parent, dict):
                    if "title" in parent:
                        if csl_type == "article-journal":
                            csl_entry["container-title"] = parent["title"]
                        elif csl_type in ["chapter"]:
                            csl_entry["container-title"] = parent["title"]

                # Handle multiple parents
                elif isinstance(parent, list):
                    for p in parent:
                        if isinstance(p, dict) and "title" in p:
                            if p.get("type", "").lower() == "periodical":
                                csl_entry["container-title"] = p["title"]
                            elif p.get("type", "").lower() in ["book", "anthology"]:
                                csl_entry["container-title"] = p["title"]

            csl_entries.append(csl_entry)

        return json.dumps(csl_entries, indent=2)

    except Exception as e:
        raise ValueError(f"Error converting to CSL-JSON: {str(e)}")


def hayagriva_to_ris(hayagriva_yaml):
    """Convert Hayagriva YAML to RIS format"""
    try:
        # Clean up the YAML before parsing
        cleaned_yaml = hayagriva_yaml.strip()

        # Parse the YAML
        try:
            data = yaml.safe_load(cleaned_yaml)
            if data is None:
                raise ValueError("Empty or invalid YAML content")
            if not isinstance(data, dict):
                raise ValueError(f"Expected a dictionary, got {type(data).__name__}")
        except Exception as yaml_error:
            print(f"YAML parsing error: {str(yaml_error)}")
            print(f"Problematic YAML: {cleaned_yaml[:100]}...")
            raise ValueError(f"Invalid YAML format: {str(yaml_error)}")

        ris_output = []

        for key, entry in data.items():
            entry_type = entry.get("type", "misc").lower()

            # Map Hayagriva types to RIS types
            type_mapping = {
                "article": "JOUR",
                "book": "BOOK",
                "chapter": "CHAP",
                "thesis": "THES",
                "report": "RPRT",
                "conference": "CONF",
                "proceedings": "CONF",
                "anthology": "BOOK",
                "web": "ELEC",
                "video": "VIDEO",
                "audio": "SOUND",
                "patent": "PAT",
                "misc": "GEN",
            }

            ris_type = type_mapping.get(entry_type, "GEN")

            # Start RIS entry
            ris_lines = [f"TY  - {ris_type}"]

            # Add ID
            ris_lines.append(f"ID  - {key}")

            # Add title
            if "title" in entry:
                ris_lines.append(f"TI  - {entry['title']}")

            # Handle author field
            if "author" in entry:
                if isinstance(entry["author"], list):
                    for author in entry["author"]:
                        ris_lines.append(f"AU  - {author}")
                else:
                    ris_lines.append(f"AU  - {entry['author']}")

            # Handle date
            if "date" in entry:
                date_value = entry["date"]
                if isinstance(date_value, str) and date_value:
                    date_components = date_value.split("-")
                    if len(date_components) >= 1:
                        ris_lines.append(f"PY  - {date_components[0]}")
                        if len(date_components) >= 2:
                            ris_lines.append(f"DA  - {date_value}")
                elif isinstance(date_value, int):
                    # If it's just a year as an integer
                    ris_lines.append(f"PY  - {date_value}")
                elif date_value is not None:
                    # Handle any other case
                    ris_lines.append(f"PY  - {date_value}")

            # Handle publisher
            if "publisher" in entry:
                if (
                    isinstance(entry["publisher"], dict)
                    and "name" in entry["publisher"]
                ):
                    ris_lines.append(f"PB  - {entry['publisher']['name']}")
                    if "location" in entry["publisher"]:
                        ris_lines.append(f"CY  - {entry['publisher']['location']}")
                else:
                    ris_lines.append(f"PB  - {entry['publisher']}")

            # Handle volume and issue
            if "volume" in entry:
                ris_lines.append(f"VL  - {entry['volume']}")

            if "issue" in entry:
                ris_lines.append(f"IS  - {entry['issue']}")

            # Handle pages
            if "page-range" in entry:
                ris_lines.append(
                    f"SP  - {entry['page-range'].split('-')[0] if '-' in entry['page-range'] else entry['page-range']}"
                )
                if "-" in entry["page-range"]:
                    ris_lines.append(f"EP  - {entry['page-range'].split('-')[1]}")

            # Handle URL
            if "url" in entry:
                if isinstance(entry["url"], dict) and "value" in entry["url"]:
                    ris_lines.append(f"UR  - {entry['url']['value']}")
                    if "date" in entry["url"]:
                        ris_lines.append(f"Y2  - {entry['url']['date']}")
                else:
                    ris_lines.append(f"UR  - {entry['url']}")

            # Handle DOI and other identifiers
            if "serial-number" in entry:
                if isinstance(entry["serial-number"], dict):
                    if "doi" in entry["serial-number"]:
                        ris_lines.append(f"DO  - {entry['serial-number']['doi']}")
                    if "isbn" in entry["serial-number"]:
                        ris_lines.append(f"SN  - {entry['serial-number']['isbn']}")
                    if "issn" in entry["serial-number"]:
                        ris_lines.append(f"SN  - {entry['serial-number']['issn']}")
                else:
                    ris_lines.append(f"M1  - {entry['serial-number']}")

            # Handle parent information
            if "parent" in entry:
                parent = entry["parent"]
                if isinstance(parent, dict):
                    if "title" in parent:
                        if ris_type == "JOUR":
                            ris_lines.append(f"JO  - {parent['title']}")
                            ris_lines.append(f"T2  - {parent['title']}")
                        elif ris_type in ["CHAP"]:
                            ris_lines.append(f"BT  - {parent['title']}")

                # Handle multiple parents
                elif isinstance(parent, list):
                    for p in parent:
                        if isinstance(p, dict) and "title" in p:
                            if p.get("type", "").lower() == "periodical":
                                ris_lines.append(f"JO  - {p['title']}")
                                ris_lines.append(f"T2  - {p['title']}")
                            elif p.get("type", "").lower() in ["book", "anthology"]:
                                ris_lines.append(f"BT  - {p['title']}")

            # End RIS entry
            ris_lines.append("ER  - ")
            ris_lines.append("")

            ris_output.append("\n".join(ris_lines))

        return "\n".join(ris_output)

    except Exception as e:
        raise ValueError(f"Error converting to RIS: {str(e)}")


@app.get("/api/citation-styles")
async def get_citation_styles():
    """Get a list of available citation styles."""
    try:
        from hayagriva_cli import get_available_styles

        styles = get_available_styles()
        return {"styles": styles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/format-references")
async def format_references(request: dict):
    """Format references in a specific citation style."""
    try:
        yaml_content = request.get("yaml_content")
        style = request.get("style")

        if not yaml_content or not style:
            raise HTTPException(status_code=400, detail="Missing yaml_content or style")

        from hayagriva_cli import format_references

        formatted_refs = format_references(yaml_content, style)

        return {"formatted_references": formatted_refs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/convert-to-hayagriva")
async def convert_to_hayagriva(input: TextInput):
    """
    Convert unstructured citation text to Hayagriva YAML format.

    This endpoint takes raw text containing citations and converts them to structured
    Hayagriva YAML format using an LLM.
    """
    try:
        # Reset progress tracker
        global conversion_progress
        conversion_progress["status"] = "splitting"
        conversion_progress["message"] = "Splitting references..."
        conversion_progress["current"] = 0
        conversion_progress["total"] = 0

        if not input.text.strip():
            raise ValueError("Input text cannot be empty")

        # Split the input text into individual references
        references = split_references(input.text)
        total_refs = len(references)
        print(f"Found {total_refs} references to process")

        # Update progress tracker
        conversion_progress["status"] = "processing"
        conversion_progress["total"] = total_refs
        conversion_progress["message"] = f"Processing {total_refs} references..."

        # Process each reference
        hayagriva_yaml_entries = []
        combined_yaml = ""

        for i, ref in enumerate(references):
            # Update progress
            current = i + 1
            conversion_progress["current"] = current
            conversion_progress["message"] = (
                f"Processing reference {current}/{total_refs}"
            )
            print(f"Processing reference {current}/{total_refs}")

            # Skip empty references
            if not ref.strip():
                continue

            # Generate a key for this reference
            ref_key = f"ref_{i+1}"

            # Prompt for the LLM with detailed instructions on Hayagriva format
            prompt = f"""Convert this citation to Hayagriva YAML format according to these rules:

1. Use the key: {ref_key}
2. Follow the Hayagriva specification with proper indentation (2 spaces)
3. Use appropriate types (web, article, book, etc.)
4. Include all available metadata (author, date, title, url, etc.)
5. DO NOT include any markdown formatting, code blocks, or backticks
6. IMPORTANT: For any title or field that contains a colon, enclose the entire value in double quotes
7. Format the YAML like this example:

ref_example:
  type: web
  title: "Example Title: With a Colon"
  author: Example Author
  url: https://example.com
  accessed: 2025-05-06

Here's the citation to convert:

{ref}

Provide ONLY the YAML output with the key, no explanations or other text."""

            try:
                # Apply rate limiting to LLM calls
                @rate_limited_api_call
                def call_llm(prompt_text):
                    return model.prompt(prompt_text)

                # Call the LLM with rate limiting
                try:
                    response = call_llm(prompt)
                    yaml_text = response.text().strip()
                except Exception as llm_error:
                    if "429" in str(llm_error) or "quota" in str(llm_error).lower():
                        print(f"Rate limit or quota exceeded: {str(llm_error)}")
                        # Add a longer delay for quota errors
                        time.sleep(2)
                        # Try again with a simpler prompt
                        simplified_prompt = f"Convert this citation to Hayagriva YAML format:\n\n{ref}\n\nYAML:"
                        try:
                            response = call_llm(simplified_prompt)
                            yaml_text = response.text().strip()
                        except Exception as retry_error:
                            raise ValueError(f"Error after retry: {str(retry_error)}")
                    else:
                        raise ValueError(f"LLM error: {str(llm_error)}")

                # Basic validation - check if it looks like YAML
                is_valid = yaml_text.startswith("{") or ":" in yaml_text

                # Add to results
                hayagriva_yaml_entries.append(
                    {"original": ref, "yaml": yaml_text, "valid": is_valid}
                )

                # Clean up the YAML text - remove any markdown code blocks or extra formatting
                yaml_text = yaml_text.replace("```yaml", "").replace("```", "").strip()

                # Fix titles with colons by adding quotes if they're not already quoted
                yaml_text = re.sub(
                    r'^(\s*title:\s*)([^"\n]*:[^\n]*)(\s*)$',
                    r'\1"\2"\3',
                    yaml_text,
                    flags=re.MULTILINE,
                )

                # Check if the YAML already has the correct key format
                if not yaml_text.startswith(f"{ref_key}:"):
                    # If the LLM included a different key or no key, fix it
                    if ":" in yaml_text.split("\n")[0]:
                        # Replace existing key with our key
                        yaml_text = yaml_text.split("\n", 1)[1].strip()

                    # Add our key and proper indentation
                    yaml_text = f"{ref_key}:\n" + "  " + yaml_text.replace("\n", "\n  ")

                # Add to combined YAML
                combined_yaml += yaml_text + "\n\n"
            except Exception as ref_error:
                print(f"Error processing reference {i+1}: {str(ref_error)}")
                hayagriva_yaml_entries.append(
                    {
                        "original": ref,
                        "yaml": f"# Error: {str(ref_error)}",
                        "valid": False,
                    }
                )

        if not hayagriva_yaml_entries:
            raise ValueError("No valid references could be processed")

        # Ensure the combined YAML is a valid Hayagriva document
        # It should be a single YAML document with all entries
        final_yaml = combined_yaml.strip()

        # Check if the YAML is valid
        try:
            yaml.safe_load(final_yaml)
            print("Generated valid YAML")
        except Exception as yaml_error:
            print(f"Warning: Generated YAML may not be valid: {str(yaml_error)}")
            # Attempt to fix common YAML issues
            try:
                # Remove any duplicate keys by keeping only the first occurrence
                seen_keys = set()
                fixed_lines = []
                current_key = None

                for line in final_yaml.split("\n"):
                    if line.strip() and not line.startswith(" ") and ":" in line:
                        # This is a top-level key
                        key = line.split(":", 1)[0].strip()
                        if key in seen_keys:
                            # Skip duplicate keys
                            current_key = None
                            continue
                        seen_keys.add(key)
                        current_key = key
                        fixed_lines.append(line)
                    elif current_key and line.strip():
                        # This is content for the current key
                        fixed_lines.append(line)
                    elif not line.strip():
                        # Empty line
                        fixed_lines.append(line)

                final_yaml = "\n".join(fixed_lines)
                print("Attempted to fix YAML by removing duplicate keys")
            except Exception as fix_error:
                print(f"Could not fix YAML: {str(fix_error)}")

        return {
            "entries": hayagriva_yaml_entries,
            "count": len(hayagriva_yaml_entries),
            "yaml": final_yaml,
        }

    except Exception as e:
        print(f"Error in convert_to_hayagriva: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


class FormatInput(BaseModel):
    yaml_content: str
    format: str = None
    output_format: str = None

    def __init__(self, **data):
        super().__init__(**data)
        # Handle both format and output_format for backward compatibility
        if self.format is None and self.output_format is not None:
            self.format = self.output_format


@app.post("/api/convert-format")
async def convert_format_endpoint(input: FormatInput):
    """Convert Hayagriva YAML to other formats."""
    try:
        # Update progress
        conversion_progress["total"] = 3
        conversion_progress["current"] = 0
        conversion_progress["message"] = "Starting conversion..."

        format_name = input.format or input.output_format
        print(f"Converting to format: {format_name}")
        print(f"YAML content preview: {input.yaml_content[:100]}...")

        # Strip any ANSI escape codes from the YAML content
        from hayagriva_cli import strip_ansi_codes

        yaml_content = strip_ansi_codes(input.yaml_content.strip())

        # Update progress
        conversion_progress["current"] = 1
        conversion_progress["message"] = "Parsing YAML..."

        # Fix titles with colons by adding quotes if they're not already quoted
        yaml_content = re.sub(
            r'^(\s*title:\s*)([^"\n]*:[^\n]*)(\s*)$',
            r'\1"\2"\3',
            yaml_content,
            flags=re.MULTILINE,
        )

        # Check if the YAML is valid
        try:
            test_parse = yaml.safe_load(yaml_content)
            if test_parse is None:
                raise ValueError("Invalid YAML: parsed as None")
            if not isinstance(test_parse, dict):
                raise ValueError(
                    f"Invalid YAML structure: expected dictionary, got {type(test_parse).__name__}"
                )
            print(f"YAML validation successful, found {len(test_parse)} entries")

            # Update progress
            format_name = input.format or input.output_format
            conversion_progress["current"] = 2
            conversion_progress["message"] = f"Converting to {format_name}..."

        except Exception as yaml_error:
            print(f"YAML validation failed: {str(yaml_error)}")

            # Try to fix by adding quotes around the entire title
            yaml_content = re.sub(
                r'^(\s*title:\s*)([^"\n]*)(\s*)$',
                r'\1"\2"\3',
                yaml_content,
                flags=re.MULTILINE,
            )

            # Try again with fixed YAML
            try:
                test_parse = yaml.safe_load(yaml_content)
                if test_parse is None:
                    raise ValueError("Invalid YAML: parsed as None after fixing")
                print(
                    f"YAML validation successful after fixing, found {len(test_parse)} entries"
                )
            except Exception as e:
                print(f"YAML validation still failed after fixing: {str(e)}")
                return {"error": f"Invalid YAML format: {str(e)}"}

            # Attempt to fix common YAML issues
            try:
                # Remove any duplicate keys by keeping only the first occurrence
                seen_keys = set()
                fixed_lines = []
                current_key = None

                for line in yaml_content.split("\n"):
                    if line.strip() and not line.startswith(" ") and ":" in line:
                        # This is a top-level key
                        key = line.split(":", 1)[0].strip()
                        if key in seen_keys:
                            # Skip duplicate keys
                            current_key = None
                            continue
                        seen_keys.add(key)
                        current_key = key
                        fixed_lines.append(line)
                    elif current_key and line.strip():
                        # This is content for the current key
                        fixed_lines.append(line)
                    elif not line.strip():
                        # Empty line
                        fixed_lines.append(line)

                yaml_content = "\n".join(fixed_lines)
                print("Attempted to fix YAML by removing duplicate keys")
            except Exception as fix_error:
                print(f"Could not fix YAML: {str(fix_error)}")

            # Try to fix the YAML by adding a dummy root key if it's missing
            if not re.match(r"^\w+:", yaml_content.split("\n")[0]):
                yaml_content = "root:\n" + "  " + yaml_content.replace("\n", "\n  ")
                print("Attempted to fix YAML by adding root key")

        # Process based on format
        result = ""

        # Import the Hayagriva CLI integration
        from hayagriva_cli import convert_with_hayagriva_cli, is_hayagriva_cli_available

        # Apply rate limiting to conversion functions
        @rate_limited_api_call
        def convert_with_rate_limit(func, content, format_type=None):
            if format_type is not None:
                return func(content, format_type)
            else:
                return func(content)

        # Check if Hayagriva CLI is available and use it if possible
        if is_hayagriva_cli_available():
            try:
                print("Using Hayagriva CLI for conversion")
                format_name = input.format or input.output_format
                result = convert_with_rate_limit(
                    convert_with_hayagriva_cli, yaml_content, format_name
                )
            except Exception as cli_error:
                print(
                    f"Hayagriva CLI error: {str(cli_error)}, falling back to Python implementation"
                )
                # Fall back to Python implementation
                format_name = input.format or input.output_format
                if format_name == "bibtex":
                    result = convert_with_rate_limit(
                        hayagriva_to_bibtex, yaml_content, None
                    )
                elif format_name == "csl-json":
                    result = convert_with_rate_limit(
                        hayagriva_to_csl_json, yaml_content, None
                    )
                elif format_name == "ris":
                    result = convert_with_rate_limit(
                        hayagriva_to_ris, yaml_content, None
                    )
                else:
                    raise ValueError(f"Unsupported output format: {format_name}")
        else:
            # Use Python implementation
            print("Using Python implementation for conversion")
            format_name = input.format or input.output_format
            if format_name == "bibtex":
                result = convert_with_rate_limit(
                    hayagriva_to_bibtex, yaml_content, None
                )
            elif format_name == "csl-json":
                result = convert_with_rate_limit(
                    hayagriva_to_csl_json, yaml_content, None
                )
            elif format_name == "ris":
                result = convert_with_rate_limit(hayagriva_to_ris, yaml_content, None)
            else:
                raise ValueError(f"Unsupported output format: {format_name}")

        # Update progress
        format_name = input.format or input.output_format
        conversion_progress["current"] = 3
        conversion_progress["status"] = "complete"
        conversion_progress["message"] = f"Conversion to {format_name} complete"

        return {"format": format_name, "converted_content": result}

    except Exception as e:
        # Update progress with error
        conversion_progress["status"] = "error"
        conversion_progress["message"] = f"Error: {str(e)}"

        print(f"Error in convert_format: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a text file containing citations.

    This endpoint accepts a text file upload and returns its content for processing.
    """
    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        file_id = str(uuid.uuid4())

        return FileUploadResponse(
            file_id=file_id, filename=file.filename, content=content_str
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.get("/api/progress")
async def get_progress():
    """Get the current progress of the conversion process"""
    return conversion_progress


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
def api_root():
    """API root endpoint that returns basic information about the service."""
    return {
        "name": "Hayagriva Converter API",
        "version": "0.1.0",
        "description": "Convert unstructured citation text to structured Hayagriva YAML and other formats",
        "endpoints": ["/api/convert-to-hayagriva", "/api/convert-format", "/upload"],
    }
