# Dataset

Raw PDFs and processed text are excluded from git (see `.gitignore`).

## Our current dataset — files in `data/raw/`

| File | Country | Report | Notes |
|------|---------|--------|-------|
| `PR149.pdf` | Ghana | DHS 2022 | Standard single volume |
| `PR157.pdf` | Nigeria | DHS 2021 | Standard single volume |
| `Final-Mini-DHS-report-FR363.pdf` | Ethiopia | DHS 2019 | Mini/abbreviated report |
| `FR380.pdf` | Kenya | DHS 2022 Vol I | Main report |
| `FR380bis.pdf` | Kenya | DHS 2022 Vol II | Secondary volume |
| `FR380erratum.pdf` | Kenya | DHS 2022 Addendum | Corrections/erratum |

## metadata.json

The `data/metadata.json` file maps each PDF filename stem to its metadata.
This is what the ingestion pipeline uses for filtered retrieval.

```json
{
  "PR149": {
    "country": "Ghana",
    "year": "2022",
    "report_type": "dhs",
    "report_title": "Ghana Demographic and Health Survey 2022"
  },
  "PR157": {
    "country": "Nigeria",
    "year": "2021",
    "report_type": "dhs",
    "report_title": "Nigeria Demographic and Health Survey 2021"
  },
  "Final-Mini-DHS-report-FR363": {
    "country": "Ethiopia",
    "year": "2019",
    "report_type": "dhs",
    "report_title": "Ethiopia Demographic and Health Survey 2019"
  },
  "FR380": {
    "country": "Kenya",
    "year": "2022",
    "report_type": "dhs",
    "report_title": "Kenya Demographic and Health Survey 2022 — Volume I"
  },
  "FR380bis": {
    "country": "Kenya",
    "year": "2022",
    "report_type": "dhs",
    "report_title": "Kenya Demographic and Health Survey 2022 — Volume II"
  },
  "FR380erratum": {
    "country": "Kenya",
    "year": "2022",
    "report_type": "dhs",
    "report_title": "Kenya Demographic and Health Survey 2022 — Addendum/Erratum"
  }
}
```

**Note on Kenya:** The three Kenya files (FR380, FR380bis, FR380erratum) all share
`country=Kenya` and `year=2022`. The ingestion pipeline treats them as separate
documents but the retriever can query them together with `WHERE country = 'Kenya'`.

## Running ingestion

```bash
# Run the full pipeline against all PDFs in data/raw/
make ingest

# With explicit metadata file
python scripts/ingest.py --metadata data/metadata.json

# Dry run — preview without writing to DB
python scripts/ingest.py --dry-run

# Drop and rebuild from scratch
python scripts/ingest.py --drop
```

## Adding more reports

1. Place the PDF in `data/raw/`
2. Add an entry to `data/metadata.json` with the filename stem as the key
3. Re-run `make ingest` — duplicate chunks are skipped via content hash
