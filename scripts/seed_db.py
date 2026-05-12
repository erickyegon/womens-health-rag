"""
Seed DB — python scripts/seed_db.py
Creates synthetic chunks for demos when real PDFs aren't available.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from rag.ingestion.indexer import VectorIndex

SEED_DOCS = [
    Document(page_content="The total fertility rate in Nigeria was 5.3 children per woman in 2021, unchanged from 2018. Urban areas show a TFR of 4.0 compared to rural areas at 5.9, indicating significant urban-rural disparities.", metadata={"country":"Nigeria","year":"2021","report_type":"dhs","report_title":"Nigeria DHS 2021","file_name":"PR157.pdf","page_number":42,"chunk_index":0}),
    Document(page_content="Contraceptive prevalence among married women in Kenya reached 61% in 2022, up from 53% in 2014. Modern method use accounts for 57%, with injectable contraceptives being the most popular method at 23%.", metadata={"country":"Kenya","year":"2022","report_type":"dhs","report_title":"Kenya DHS 2022 Vol I","file_name":"FR380.pdf","page_number":87,"chunk_index":0}),
    Document(page_content="In Ghana, skilled birth attendance reached 84% in 2022, a significant increase from 74% in 2014. The proportion of births delivered in health facilities increased to 79%. Regional disparities persist with Northern Region at 52%.", metadata={"country":"Ghana","year":"2022","report_type":"dhs","report_title":"Ghana DHS 2022","file_name":"PR149.pdf","page_number":63,"chunk_index":0}),
    Document(page_content="Child stunting in Ethiopia affected 37% of children under 5 in 2019. Wasting prevalence stood at 7%. Stunting is highest in children aged 24-35 months. Children of mothers with no education show stunting rates of 43% compared to 25% for mothers with secondary or higher education.", metadata={"country":"Ethiopia","year":"2019","report_type":"dhs","report_title":"Ethiopia DHS 2019","file_name":"Final-Mini-DHS-report-FR363.pdf","page_number":34,"chunk_index":0}),
    Document(page_content="The under-5 mortality rate in Nigeria declined from 128 per 1,000 live births in 2013 to 117 in 2021. Neonatal mortality accounts for 43% of under-5 deaths. North West zone shows the highest under-5 mortality at 185 per 1,000.", metadata={"country":"Nigeria","year":"2021","report_type":"dhs","report_title":"Nigeria DHS 2021","file_name":"PR157.pdf","page_number":78,"chunk_index":1}),
    Document(page_content="In Kenya, the maternal mortality ratio was estimated at 362 per 100,000 live births during the 2017-2022 period. Postpartum haemorrhage and hypertensive disorders are leading causes. Counties in arid and semi-arid regions show higher ratios.", metadata={"country":"Kenya","year":"2022","report_type":"dhs","report_title":"Kenya DHS 2022 Vol I","file_name":"FR380.pdf","page_number":112,"chunk_index":1}),
]

print("Seeding database with sample documents...")
idx = VectorIndex()
idx.init_schema()
n = idx.upsert_documents(SEED_DOCS)
print(f"Seeded {n} documents. Total in DB: {idx.count()}")
