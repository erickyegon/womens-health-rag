"""
Multimodal Loader — Episode 2B (upgrade to Episode 2)
=======================================================
Production-grade PDF ingestion using a THREE-LAYER strategy:

LAYER 1 — Docling (IBM Research)
    - DocLayNet model: understands document layout (headers, paragraphs, tables, figures)
    - TableFormer model: reconstructs table structure as Markdown
    - Outputs: structured text + Markdown tables + image coordinates
    - Best for: text + table extraction, FREE, runs locally, LangChain native

LAYER 2 — GPT-4o Vision (OpenAI)
    - For every figure/chart/image Docling identifies, pass the cropped image to GPT-4o
    - GPT-4o reads bar charts, line graphs, maps, infographics as natural language
    - Outputs: rich text description of what the visual shows, with data points
    - Best for: Figure 2's stacked bar chart, choropleth maps, trend lines

LAYER 3 — Table-to-Prose (deterministic)
    - Docling gives us Markdown tables → we ALSO generate natural language prose
    - Prose version is stored alongside Markdown for semantic retrieval
    - "TFR Nigeria Urban 3.9, Rural 5.6, Total 4.8" → searchable sentences
    - Best for: ensuring tables are retrieved for semantic queries, not just exact ones

WHY THIS MATTERS FOR DHS REPORTS:
    The Nigeria PR157 report contains:
    - Table 3: TFR by age group × residence (urban/rural/total) — Layer 1 + 3
    - Figure 1: Line chart showing TFR trends 2008–2024 — Layer 2
    - Figure 2: Stacked bar chart of family planning demand — Layer 2
    - Definition boxes: Bold term + paragraph text — Layer 1 preserves structure
    - Multi-column layouts — DocLayNet handles reading order correctly

RAGAS IMPACT:
    Without multimodal: "What is the TFR in rural Nigeria?" → MISS (table was garbled)
    With multimodal:    "What is the TFR in rural Nigeria?" → HIT (5.6, Table 3, p.42)

    Without multimodal: "Show the trend in contraceptive use in Nigeria" → MISS (Figure 2 invisible)
    With multimodal:    "Show the trend in contraceptive use" → HIT (GPT-4o description retrieved)

COST ESTIMATE FOR DHS CORPUS:
    Docling:        $0.00 (local, CPU-only)
    GPT-4o vision:  ~$0.003 per image × ~200 figures = ~$0.60 total
    Total upgrade:  < $1 for the entire 6-report corpus

Episode 2B walkthrough:
    1. Show standard loader output on Table 3 — garbled
    2. Show Docling output on same page — perfect Markdown table
    3. Show Figure 2 — invisible to text extractors
    4. Show GPT-4o vision description of Figure 2 — full data captured
    5. Show table-to-prose conversion — makes tables semantically searchable
    6. Re-run ingestion with multimodal loader
    7. Query "TFR rural Nigeria" — show the hit that was previously a miss
"""

from __future__ import annotations

import base64
import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Element types ─────────────────────────────────────────────────────────────

class ElementType(str, Enum):
    TEXT    = "text"
    TABLE   = "table"
    FIGURE  = "figure"
    HEADING = "heading"
    LIST    = "list"


@dataclass
class DocElement:
    """
    A single structural element extracted from a PDF page.
    Tables, figures, headings, and text paragraphs are each separate elements.
    This is the key improvement over the old RawPage approach which merged everything.
    """
    element_type: ElementType
    content:      str                  # text / Markdown table / figure description
    page_number:  int
    source_file:  str
    file_name:    str
    total_pages:  int

    # Domain metadata
    country:      str = ""
    year:         str = ""
    report_type:  str = ""
    report_title: str = ""

    # Element-specific metadata
    table_markdown:  str | None = None   # for TABLE: raw Markdown from Docling
    table_prose:     str | None = None   # for TABLE: LLM-generated prose summary
    figure_caption:  str | None = None   # for FIGURE: caption text if found
    figure_base64:   str | None = None   # for FIGURE: base64 image (not indexed)
    heading_level:   int | None = None   # for HEADING: 1=H1, 2=H2, etc.
    confidence:      float = 1.0         # extraction confidence

    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document for indexing."""
        return Document(
            page_content=self.content,
            metadata={
                "element_type":  self.element_type.value,
                "source":        self.source_file,
                "file_name":     self.file_name,
                "page_number":   self.page_number,
                "total_pages":   self.total_pages,
                "country":       self.country,
                "year":          self.year,
                "report_type":   self.report_type,
                "report_title":  self.report_title,
                "has_table":     self.element_type == ElementType.TABLE,
                "has_figure":    self.element_type == ElementType.FIGURE,
                "figure_caption": self.figure_caption,
                "confidence":    self.confidence,
            }
        )

    @property
    def is_empty(self) -> bool:
        return len(self.content.strip()) < 20


# ── Main multimodal loader ────────────────────────────────────────────────────

class MultimodalLoader:
    """
    Three-layer multimodal PDF loader.

    Usage:
        loader = MultimodalLoader(vision_enabled=True)
        elements = loader.load_pdf(
            Path("data/raw/PR157.pdf"),
            country="Nigeria", year="2021",
        )
        # elements: list[DocElement] — one per structural element
        # Convert to Documents: [e.to_langchain_document() for e in elements]

    Episode 2B teaching order:
        1. Start with vision_enabled=False — show Docling alone (tables fixed)
        2. Enable vision_enabled=True — show Figure 2 suddenly described
        3. Show RAGAS improvement on the table query
    """

    def __init__(
        self,
        vision_enabled: bool = True,
        vision_model:   str  = "gpt-4o",
        table_prose:    bool = True,
        batch_vision:   int  = 5,
    ):
        self.vision_enabled = vision_enabled
        self.vision_model   = vision_model
        self.table_prose    = table_prose
        self.batch_vision   = batch_vision
        self._vision_client = None   # lazy init

    def load_pdf(
        self,
        path: Path | str,
        country:      str = "",
        year:         str = "",
        report_type:  str = "dhs",
        report_title: str = "",
    ) -> list[DocElement]:
        """
        Load a PDF and return structured DocElements.

        Args:
            path:         PDF file path
            country:      e.g. "Nigeria"
            year:         e.g. "2021"
            report_type:  "dhs" | "status_of_women" | "other"
            report_title: Full human-readable title

        Returns:
            List of DocElement objects, one per structural element.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Multimodal loading: %s", path.name)

        meta = {
            "source_file":  str(path.resolve()),
            "file_name":    path.name,
            "country":      country,
            "year":         year,
            "report_type":  report_type,
            "report_title": report_title or path.stem,
        }

        # Layer 1: Docling extraction
        elements = self._docling_extract(path, meta)

        # Layer 2: GPT-4o vision for figures
        if self.vision_enabled:
            elements = self._vision_describe_figures(elements, path)

        # Layer 3: Table-to-prose conversion
        if self.table_prose:
            elements = self._generate_table_prose(elements)

        non_empty = [e for e in elements if not e.is_empty]
        logger.info(
            "Loaded %d elements (%d text, %d tables, %d figures) from %s",
            len(non_empty),
            sum(1 for e in non_empty if e.element_type == ElementType.TEXT),
            sum(1 for e in non_empty if e.element_type == ElementType.TABLE),
            sum(1 for e in non_empty if e.element_type == ElementType.FIGURE),
            path.name,
        )
        return non_empty

    def load_directory(
        self,
        directory: Path | str,
        metadata_map: dict[str, dict] | None = None,
        glob: str = "**/*.pdf",
    ) -> list[DocElement]:
        """Load all PDFs in a directory."""
        directory = Path(directory)
        pdfs = sorted(directory.glob(glob))
        all_elements: list[DocElement] = []

        for pdf in pdfs:
            meta = (metadata_map or {}).get(pdf.stem, {})
            elements = self.load_pdf(
                pdf,
                country=meta.get("country", ""),
                year=meta.get("year", ""),
                report_type=meta.get("report_type", "dhs"),
                report_title=meta.get("report_title", ""),
            )
            all_elements.extend(elements)

        logger.info(
            "Directory load complete: %d elements from %d PDFs",
            len(all_elements), len(pdfs),
        )
        return all_elements

    def to_documents(self, elements: list[DocElement]) -> list[Document]:
        """
        Convert DocElements to LangChain Documents for indexing.

        For tables: creates TWO documents — Markdown version + prose version.
        This gives both exact structural retrieval AND semantic retrieval.
        """
        docs: list[Document] = []
        for elem in elements:
            # Primary document
            docs.append(elem.to_langchain_document())

            # For tables: also index the prose version separately
            if (elem.element_type == ElementType.TABLE
                    and elem.table_prose
                    and elem.table_prose != elem.content):
                prose_doc = Document(
                    page_content=elem.table_prose,
                    metadata={
                        **elem.to_langchain_document().metadata,
                        "element_type": "table_prose",
                        "source_element": "table",
                    }
                )
                docs.append(prose_doc)

        return docs

    # ── Layer 1: Docling ──────────────────────────────────────────────────────

    def _docling_extract(self, path: Path, meta: dict) -> list[DocElement]:
        """
        Use Docling to extract structured elements from a PDF.
        DocLayNet identifies: headings, paragraphs, tables, figures, lists.
        TableFormer reconstructs table structure as Markdown.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
        except ImportError:
            logger.warning(
                "Docling not installed — falling back to PyMuPDF. "
                "Run: uv add docling"
            )
            return self._pymupdf_fallback(path, meta)

        # Configure pipeline — save images for vision processing later
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.images_scale = 2.0          # high-res for vision model
        pipeline_opts.generate_page_images = False
        pipeline_opts.generate_picture_images = True  # extract figure images

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

        logger.info("Running Docling on %s...", path.name)
        result = converter.convert(str(path))
        doc    = result.document

        # Count pages
        total_pages = len(doc.pages) if hasattr(doc, 'pages') else 1

        elements: list[DocElement] = []

        # ── Extract text elements ─────────────────────────────────────────────
        for item, level in doc.iterate_items():
            item_type = type(item).__name__

            if item_type in ("TextItem", "ParagraphItem"):
                elements.append(DocElement(
                    element_type=ElementType.TEXT,
                    content=item.text,
                    page_number=self._get_page(item, doc),
                    total_pages=total_pages,
                    **meta,
                ))

            elif item_type == "SectionHeaderItem":
                elements.append(DocElement(
                    element_type=ElementType.HEADING,
                    content=item.text,
                    page_number=self._get_page(item, doc),
                    total_pages=total_pages,
                    heading_level=getattr(item, 'level', 1),
                    **meta,
                ))

            elif item_type == "TableItem":
                md_table = item.export_to_markdown()
                elements.append(DocElement(
                    element_type=ElementType.TABLE,
                    content=md_table,
                    table_markdown=md_table,
                    page_number=self._get_page(item, doc),
                    total_pages=total_pages,
                    confidence=0.95,
                    **meta,
                ))

            elif item_type in ("PictureItem", "FigureItem"):
                caption = ""
                if hasattr(item, 'caption') and item.caption:
                    caption = str(item.caption)
                # Extract image bytes for vision processing
                img_b64 = self._extract_figure_image(item, result)
                elements.append(DocElement(
                    element_type=ElementType.FIGURE,
                    content=caption or f"[Figure on page {self._get_page(item, doc)}]",
                    figure_caption=caption,
                    figure_base64=img_b64,
                    page_number=self._get_page(item, doc),
                    total_pages=total_pages,
                    confidence=0.5,  # low until vision processes it
                    **meta,
                ))

            elif item_type == "ListItem":
                elements.append(DocElement(
                    element_type=ElementType.LIST,
                    content=item.text,
                    page_number=self._get_page(item, doc),
                    total_pages=total_pages,
                    **meta,
                ))

        logger.info(
            "Docling extracted %d elements from %s",
            len(elements), path.name,
        )
        return elements

    # ── Layer 2: GPT-4o Vision ────────────────────────────────────────────────

    def _vision_describe_figures(
        self, elements: list[DocElement], path: Path
    ) -> list[DocElement]:
        """
        For each FIGURE element, use GPT-4o to generate a rich text description.

        If no base64 image is available from Docling (e.g. complex figure),
        we fall back to rendering the page with PyMuPDF and cropping.

        The description becomes the figure's content — making it searchable.
        """
        figures = [e for e in elements if e.element_type == ElementType.FIGURE]
        if not figures:
            return elements

        logger.info("Processing %d figures with GPT-4o vision...", len(figures))

        # Lazy-init vision client
        if not self._vision_client:
            self._vision_client = self._init_vision_client()

        for fig in figures:
            if not fig.figure_base64:
                # Try to render the page and extract figure region
                fig.figure_base64 = self._render_page_image(path, fig.page_number)

            if fig.figure_base64:
                description = self._gpt4o_describe(
                    fig.figure_base64,
                    context=fig.figure_caption or "",
                    report_context=f"{fig.country} {fig.report_title} {fig.year}",
                )
                fig.content    = description
                fig.confidence = 0.9
            else:
                logger.warning(
                    "Could not extract image for figure on page %d of %s",
                    fig.page_number, fig.file_name,
                )

        return elements

    def _gpt4o_describe(
        self, image_b64: str, context: str = "", report_context: str = ""
    ) -> str:
        """
        Send a figure image to GPT-4o and get a structured description.

        The prompt is specific to DHS health reports — asking for:
        - Chart type and what it shows
        - All data values visible
        - Trends and patterns
        - Connection to the surrounding text
        """
        prompt = f"""You are analysing a figure from a Demographic and Health Survey (DHS) report.
{f'Report context: {report_context}' if report_context else ''}
{f'Figure caption: {context}' if context else ''}

Describe this figure comprehensively for a health data retrieval system:

1. CHART TYPE: What kind of visualisation is this? (bar chart, line graph, map, table, etc.)
2. TITLE/SUBJECT: What does it show? Include all axis labels and legend items.
3. DATA VALUES: List ALL specific numbers, percentages, and values visible.
4. TIME PERIOD: What years or periods are shown?
5. BREAKDOWN: What categories, groups, or regions are compared?
6. KEY TRENDS: What are the main patterns or findings?
7. CONTEXT: How does this connect to maternal health, fertility, contraception, or child health?

Be specific and include all numerical values — this description will be used to answer 
precise data queries about women's health statistics."""

        try:
            response = self._vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",   # high detail for charts with small numbers
                            }
                        }
                    ]
                }],
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("GPT-4o vision call failed: %s", e)
            return f"[Figure — vision processing failed: {e}]"

    # ── Layer 3: Table-to-Prose ───────────────────────────────────────────────

    def _generate_table_prose(self, elements: list[DocElement]) -> list[DocElement]:
        """
        Convert Markdown tables to natural language prose.

        WHY: Vector embeddings of Markdown tables are poor.
        "| 3.9 | 5.6 | 4.8 |" is not semantically close to
        "total fertility rate urban rural national".

        We generate prose that IS semantically close to likely queries.

        Example:
            Table 3 (Markdown):
                | Age group | Urban | Rural | Total |
                | TFR(15-49)| 3.9   | 5.6   | 4.8   |

            Table 3 (Prose):
                Table 3 shows current fertility rates in Nigeria by residence.
                The total fertility rate (TFR) for women aged 15-49 is 3.9 in
                urban areas, 5.6 in rural areas, and 4.8 nationally.
                The general fertility rate (GFR) is 129 urban, 190 rural, 160 total.
                The crude birth rate (CBR) is 28 urban, 38 rural, 33 total.
        """
        table_elements = [e for e in elements if e.element_type == ElementType.TABLE]
        if not table_elements:
            return elements

        logger.info("Generating prose for %d tables...", len(table_elements))

        for table in table_elements:
            if not table.table_markdown:
                continue
            prose = self._table_markdown_to_prose(
                table.table_markdown,
                report_context=f"{table.country} {table.report_title} {table.year}",
                page=table.page_number,
            )
            table.table_prose = prose

        return elements

    def _table_markdown_to_prose(
        self, markdown: str, report_context: str = "", page: int = 0
    ) -> str:
        """
        Convert a Markdown table to descriptive prose using GPT-4o-mini.
        Uses the cheap mini model — tables are structured, not visually complex.
        """
        if not self._vision_client:
            self._vision_client = self._init_vision_client()

        prompt = f"""Convert this Markdown table from a DHS health report into clear, 
descriptive prose that captures ALL the data values.

Report context: {report_context} (page {page})

Table:
{markdown}

Write 2-5 sentences that:
1. State what the table shows (its subject)
2. Include ALL specific numerical values with their row/column context
3. Use natural language that matches how a health researcher would ask about this data
4. Include any footnotes or notes visible in the table

Example style: "Table 3 shows fertility rates by residence in Nigeria (2021). 
The total fertility rate for women aged 15-49 is 3.9 urban, 5.6 rural, and 4.8 nationally.
General fertility rates are 129, 190, and 160 per 1,000 women aged 15-44 respectively."

Output ONLY the prose description, nothing else."""

        try:
            from rag.config.settings import get_settings
            settings = get_settings()
            response = self._vision_client.chat.completions.create(
                model="gpt-4o-mini",   # cheap — tables are structured text not images
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Table prose generation failed: %s", e)
            return markdown  # fall back to markdown

    # ── Helper methods ────────────────────────────────────────────────────────

    def _init_vision_client(self):
        try:
            import openai
            from rag.config.settings import get_settings
            s = get_settings()
            return openai.OpenAI(api_key=s.openai_api_key.get_secret_value())
        except Exception as e:
            logger.error("Could not init OpenAI client: %s", e)
            raise

    def _get_page(self, item: Any, doc: Any) -> int:
        """Extract page number from a Docling item."""
        try:
            if hasattr(item, 'prov') and item.prov:
                return item.prov[0].page_no
        except Exception:
            pass
        return 1

    def _extract_figure_image(self, item: Any, result: Any) -> str | None:
        """Extract a figure's image bytes from Docling as base64."""
        try:
            if hasattr(item, 'image') and item.image:
                img_data = item.image.pil_image
                import io
                buf = io.BytesIO()
                img_data.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.debug("Could not extract figure image: %s", e)
        return None

    def _render_page_image(self, path: Path, page_number: int) -> str | None:
        """
        Render an entire page as a high-res image using PyMuPDF.
        Fallback when Docling can't extract the figure image directly.
        Used for full-page visual processing.
        """
        try:
            import pymupdf
            doc  = pymupdf.open(str(path))
            page = doc[page_number - 1]
            pix  = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            doc.close()
            return base64.b64encode(img_bytes).decode()
        except Exception as e:
            logger.warning("Page render failed for page %d: %s", page_number, e)
            return None

    def _pymupdf_fallback(self, path: Path, meta: dict) -> list[DocElement]:
        """
        Fallback extractor when Docling is not installed.
        Returns basic text elements without table/figure understanding.
        """
        try:
            import pymupdf
        except ImportError:
            raise ImportError("Neither Docling nor PyMuPDF is installed.")

        logger.warning("Using PyMuPDF fallback — install Docling for full multimodal support")
        doc   = pymupdf.open(str(path))
        total = len(doc)
        elements: list[DocElement] = []

        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) > 50:
                elements.append(DocElement(
                    element_type=ElementType.TEXT,
                    content=text,
                    page_number=i,
                    total_pages=total,
                    confidence=0.5,
                    **meta,
                ))

        doc.close()
        return elements


# ── Convenience functions ─────────────────────────────────────────────────────

def load_pdf_multimodal(
    path: Path | str,
    country: str = "",
    year: str = "",
    report_type: str = "dhs",
    report_title: str = "",
    vision_enabled: bool = True,
) -> list[Document]:
    """
    One-call multimodal PDF loader returning LangChain Documents.

    Usage (Episode 2B notebook):
        docs = load_pdf_multimodal(
            "data/raw/PR157.pdf",
            country="Nigeria", year="2021",
        )
        print(f"Documents: {len(docs)}")
        tables = [d for d in docs if d.metadata['has_table']]
        figures = [d for d in docs if d.metadata['has_figure']]
    """
    loader   = MultimodalLoader(vision_enabled=vision_enabled)
    elements = loader.load_pdf(path, country=country, year=year,
                               report_type=report_type, report_title=report_title)
    return loader.to_documents(elements)


def load_directory_multimodal(
    directory: Path | str,
    metadata_map: dict | None = None,
    vision_enabled: bool = True,
) -> list[Document]:
    """
    Load all PDFs in a directory with multimodal extraction.
    Drop-in replacement for the Episode 1–2 load_directory() + chunk_pages() flow.
    """
    loader   = MultimodalLoader(vision_enabled=vision_enabled)
    elements = loader.load_directory(directory, metadata_map=metadata_map)
    return loader.to_documents(elements)


def element_stats(elements: list[DocElement]) -> dict:
    """Summary statistics for a multimodal load — used in Episode 2B notebook."""
    from collections import Counter
    type_counts = Counter(e.element_type.value for e in elements)
    return {
        "total":      len(elements),
        "text":       type_counts.get("text", 0),
        "tables":     type_counts.get("table", 0),
        "figures":    type_counts.get("figure", 0),
        "headings":   type_counts.get("heading", 0),
        "lists":      type_counts.get("list", 0),
        "with_vision": sum(1 for e in elements
                          if e.element_type == ElementType.FIGURE
                          and e.confidence > 0.5),
        "with_prose":  sum(1 for e in elements
                          if e.element_type == ElementType.TABLE
                          and e.table_prose is not None),
    }
