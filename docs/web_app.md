# Web Interface Module

**File**: `web/app.py`

## Overview

The web interface provides a Flask-based dashboard for browsing papers, viewing analysis results, and exporting reports. It serves as the user-facing layer of the Journal Club pipeline.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOURNAL_CLUB_WEB_PORT` | `5000` | Web server port |
| `JOURNAL_CLUB_LITERATURE_MEMORY_PATH` | `literature_memory.json` | Path to literature memory |

### Flask Configuration

- Debug mode: Disabled by default
- Template folder: `web/templates`
- Static folder: `web/static`

## Routes

### `/`

**GET** - Main dashboard

**Template:** `index.html`

**Context:**
- `stats`: Overall statistics from memory
- `topics`: List of topics with paper counts
- `domains`: List of domains with paper counts

**Example:**
```python
GET http://localhost:5000/
```

### `/topic/<topic_name>`

**GET** - Topic-specific view

**Template:** `topic_view.html`

**Parameters:**
- `topic_name`: Topic name (URL-encoded)

**Context:**
- `topic`: Topic name
- `papers`: List of papers for the topic
- `stats`: Topic-specific statistics

**Example:**
```python
GET http://localhost:5000/topic/RNA-Protein%20Interactions
```

### `/paper/<paper_id>`

**GET** - Paper detail view

**Template:** `paper_detail.html`

**Parameters:**
- `paper_id`: Paper DOI or title (URL-encoded)

**Context:**
- `paper`: Paper dict with full analysis
- `recommendations`: Recommendations for the paper

**Example:**
```python
GET http://localhost:5000/paper/10.1234%2Fexample.2024.001
```

### `/api/stats`

**GET** - API endpoint for statistics

**Response:** JSON
```json
{
  "total_papers": 100,
  "analyzed_papers": 75,
  "by_topic": {"Topic1": 50, "Topic2": 50},
  "by_domain": {"biophysics": 100}
}
```

**Example:**
```python
GET http://localhost:5000/api/stats
```

### `/api/papers`

**GET** - API endpoint for papers

**Query Parameters:**
- `topic`: Filter by topic (optional)
- `domain`: Filter by domain (optional)
- `limit`: Maximum number of papers (default: 100)

**Response:** JSON array of paper dicts

**Example:**
```python
GET http://localhost:5000/api/papers?topic=RNA-Protein%20Interactions&limit=50
```

### `/api/paper/<paper_id>`

**GET** - API endpoint for single paper

**Parameters:**
- `paper_id`: Paper DOI or title

**Response:** JSON paper dict

**Example:**
```python
GET http://localhost:5000/api/paper/10.1234%2Fexample.2024.001
```

### `/export/topic/<topic_name>`

**GET** - Export topic report as markdown

**Parameters:**
- `topic_name`: Topic name

**Response:** Markdown file download

**Example:**
```python
GET http://localhost:5000/export/topic/RNA-Protein%20Interactions
```

### `/export/paper/<paper_id>`

**GET** - Export paper report as markdown

**Parameters:**
- `paper_id`: Paper DOI or title

**Response:** Markdown file download

**Example:**
```python
GET http://localhost:5000/export/paper/10.1234%2Fexample.2024.001
```

### `/export/summary`

**GET** - Export summary report as markdown

**Response:** Markdown file download

**Example:**
```python
GET http://localhost:5000/export/summary
```

### `/export/json/<topic_name>`

**GET** - Export topic data as JSON

**Parameters:**
- `topic_name`: Topic name

**Response:** JSON file download

**Example:**
```python
GET http://localhost:5000/export/json/RNA-Protein%20Interactions
```

## Templates

### `index.html`

Main dashboard template.

**Features:**
- Overall statistics display
- Topic list with paper counts
- Domain list with paper counts
- Export buttons for summary report
- Links to topic views

**Sections:**
- Header with title
- Statistics cards
- Topics table
- Domains table
- Export section

### `topic_view.html`

Topic-specific view template.

**Features:**
- Topic name and statistics
- Paper list with:
  - Title and authors
  - Year and DOI
  - Summary preview
  - Quality score indicator
  - Gap analysis tags
  - Link to paper detail
- Export buttons for topic

**Sections:**
- Header with topic name
- Statistics bar
- Paper cards
- Export section

### `paper_detail.html`

Paper detail view template.

**Features:**
- Full paper metadata
- Abstract display
- Summary section
- Gap analysis with categories
- Quality scores with indicators
- Critique section
- Recommendation type
- Related papers list
- Export button for paper

**Sections:**
- Header with title and metadata
- Abstract section
- Summary section
- Gap analysis section
- Quality scores section
- Critique section
- Recommendations section
- Related papers section
- Export section

## Helper Functions

### `get_memory()`

Get or create memory instance.

**Returns:**
- JournalClubMemory instance

**Behavior:**
- Uses cached instance if available
- Creates new instance if not cached
- Loads from `JOURNAL_CLUB_LITERATURE_MEMORY_PATH`

### `format_score(score)`

Format quality score for display.

**Parameters:**
- `score`: Quality score (0-1)

**Returns:**
- Formatted string with color indicator

**Indicators:**
- 🟢 High quality (≥ 0.8)
- 🟡 Medium quality (0.6 - 0.8)
- 🔴 Low quality (< 0.6)

### `format_gaps(gap_analysis)`

Format gap analysis for display.

**Parameters:**
- `gap_analysis`: Gap analysis dict

**Returns:**
- List of formatted gap strings

**Behavior:**
- Flattens gap categories
- Returns list of gap descriptions

### `safe_url_encode(text)`

URL-encode text for use in URLs.

**Parameters:**
- `text`: Text to encode

**Returns:**
- URL-encoded string

## Usage Patterns

### Start Web Server
```bash
./scripts/run_journal_club.sh web
```

### Access Dashboard
```
http://localhost:5000
```

### Browse Topics
```
http://localhost:5000/topic/RNA-Protein%20Interactions
```

### View Paper
```
http://localhost:5000/paper/10.1234%2Fexample.2024.001
```

### Export Reports
```
http://localhost:5000/export/topic/RNA-Protein%20Interactions
http://localhost:5000/export/paper/10.1234%2Fexample.2024.001
http://localhost:5000/export/summary
```

### API Usage
```python
import requests

# Get statistics
stats = requests.get("http://localhost:5000/api/stats").json()

# Get papers
papers = requests.get("http://localhost:5000/api/papers?topic=My Topic").json()

# Get single paper
paper = requests.get("http://localhost:5000/api/paper/10.1234/example").json()
```

## Error Handling

- Missing memory file: Returns empty statistics
- Invalid paper ID: Returns 404
- Missing analysis fields: Displays as "Not analyzed"
- Template errors: Logged and return 500

## Performance Considerations

- Memory is loaded once per request (consider caching)
- Large topic pages may be slow (add pagination)
- API endpoints are synchronous
- File exports are generated on-demand

## Security Considerations

- No authentication (add for production)
- No rate limiting (add for production)
- No input validation (add for production)
- File paths are not sanitized (add for production)

## Customization

### Add Authentication
```python
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == 'admin' and password == 'secret'

@app.route('/')
@auth.login_required
def index():
    ...
```

### Add Pagination
```python
@app.route('/topic/<topic_name>')
def topic_view(topic_name):
    page = request.args.get('page', 1, type=int)
    per_page = 20
    papers = memory Papers_by_topic(topic_name, limit=per_page, offset=(page-1)*per_page)
    ...
```

### Add Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
@cache.cached(timeout=300)
def index():
    ...
```

## Troubleshooting

### Server won't start
- Check port is not in use
- Verify Flask is installed
- Check memory file path

### Pages load slowly
- Reduce papers per page
- Add pagination
- Add caching
- Optimize memory queries

### API returns errors
- Check memory file exists
- Verify paper IDs are valid
- Check query parameters

### Exports fail
- Check output directory permissions
- Verify report generator is working
- Check disk space
