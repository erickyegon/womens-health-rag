# Dataset

This directory contains the raw and processed data for the Women's Health RAG system.

## Why data/ is not committed

Raw PDFs and processed text are excluded from git (see `.gitignore`).
Reasons:
- File size (DHS reports are 200–500MB each)
- Licensing (DHS data requires registration)
- Reproducibility — the ingestion pipeline recreates processed/ from raw/

## Downloading the data

### Demographic and Health Surveys (DHS)

1. Register at https://dhsprogram.com/data/available-datasets.cfm (free)
2. After approval, download the survey final reports (PDF format)
3. Place PDFs in `data/raw/` with this naming convention:
   `{country_iso3}_{survey_year}_dhs.pdf`
   Example: `NGA_2021_dhs.pdf`

**Recommended starting set** (all freely available after registration):
| Country   | Year | Filename              |
|-----------|------|-----------------------|
| Nigeria   | 2021 | NGA_2021_dhs.pdf      |
| Kenya     | 2022 | KEN_2022_dhs.pdf      |
| Ghana     | 2022 | GHA_2022_dhs.pdf      |
| Ethiopia  | 2019 | ETH_2019_dhs.pdf      |
| Tanzania  | 2022 | TZA_2022_dhs.pdf      |

### Status of Women Reports

Download from: https://statusofwomendata.org/

Place PDFs in `data/raw/` with naming: `status_of_women_{year}.pdf`

## Metadata file

Create `data/metadata.json` mapping filename stems to metadata.
This is used by the ingestion pipeline for filtered retrieval.

```json
{
  "NGA_2021_dhs": {
    "country": "Nigeria",
    "year": "2021",
    "report_type": "dhs",
    "report_title": "Nigeria Demographic and Health Survey 2021"
  },
  "KEN_2022_dhs": {
    "country": "Kenya",
    "year": "2022",
    "report_type": "dhs",
    "report_title": "Kenya Demographic and Health Survey 2022"
  }
}
```

## Running ingestion

Once PDFs are in `data/raw/` and metadata.json is ready:

```bash
# With metadata
make ingest

# Or directly
python scripts/ingest.py --metadata data/metadata.json

# Dry run to preview (no DB writes)
python scripts/ingest.py --dry-run
```
