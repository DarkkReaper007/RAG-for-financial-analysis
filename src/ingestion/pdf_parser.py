"""
Stage 1: PDF Parsing & Ingestion

Structure-aware PDF parser using PyMuPDF (fitz).
Extracts text blocks with font metadata, detects headers via font-size heuristics,
and identifies table regions. Preserves SEBI-mandated document hierarchy.
"""

import fitz  # PyMuPDF
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A single extracted text block from a PDF page."""
    text: str
    page_number: int
    font_size: float
    is_bold: bool
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    block_type: str = "text"  # "text", "header", "table"


@dataclass
class ParsedSection:
    """A logical section of the document with its header and content blocks."""
    title: str
    level: int  # 1 = top-level, 2 = sub-section, etc.
    page_start: int
    page_end: int
    blocks: List[TextBlock] = field(default_factory=list)
    raw_text: str = ""

    def get_full_text(self) -> str:
        """Get the full text of this section."""
        if self.raw_text:
            return self.raw_text
        return "\n".join(b.text for b in self.blocks if b.text.strip())


@dataclass
class ParsedDocument:
    """Complete parsed representation of an IPO prospectus."""
    filename: str
    company_name: str
    total_pages: int
    sections: List[ParsedSection] = field(default_factory=list)
    tables: List[TextBlock] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class PDFParser:
    """
    Structure-aware PDF parser for Indian IPO prospectuses.
    
    Uses font-size heuristics to detect section headers and builds
    a hierarchical document structure that preserves the SEBI-mandated
    document layout.
    """

    # Common IPO prospectus section titles (for detection)
    KNOWN_SECTIONS = [
        "risk factors", "business overview", "objects of the issue",
        "basis for issue price", "financial information", "legal and other information",
        "general information", "capital structure", "management",
        "promoter", "dividend policy", "industry overview",
        "outstanding litigation", "government approvals", "other regulatory",
        "material contracts", "financial statements", "auditor",
        "about the company", "key managerial personnel", "related party",
        "stock market data", "accounting ratios", "offer structure",
        "terms of the issue", "issue procedure", "main provisions",
        "restrictions on transfer", "book building process",
    ]

    def __init__(self, min_header_font_size: float = 11.0):
        """
        Args:
            min_header_font_size: Minimum font size to consider as a header.
                                  Adjusted dynamically per document.
        """
        self.min_header_font_size = min_header_font_size

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file into a structured ParsedDocument.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            ParsedDocument with hierarchical sections.
        """
        path = Path(pdf_path)
        logger.info(f"Parsing PDF: {path.name}")

        doc = fitz.open(str(path))
        total_pages = len(doc)
        logger.info(f"  Total pages: {total_pages}")

        # Phase 1: Extract all text blocks with font metadata
        all_blocks = self._extract_blocks(doc)
        logger.info(f"  Extracted {len(all_blocks)} text blocks")

        # Phase 2: Determine font size thresholds dynamically
        font_sizes = [b.font_size for b in all_blocks if b.font_size > 0]
        if font_sizes:
            sorted_sizes = sorted(set(font_sizes), reverse=True)
            # Top 3 unique font sizes are likely headers
            header_thresholds = sorted_sizes[:min(4, len(sorted_sizes))]
            body_size = sorted_sizes[min(4, len(sorted_sizes) - 1)] if len(sorted_sizes) > 4 else sorted_sizes[-1]
        else:
            header_thresholds = [14.0, 12.0]
            body_size = 10.0

        logger.info(f"  Header font sizes detected: {header_thresholds}")
        logger.info(f"  Body font size: {body_size}")

        # Phase 3: Classify blocks as headers vs body text
        self._classify_blocks(all_blocks, header_thresholds, body_size)

        # Phase 4: Extract tables
        tables = self._extract_tables(doc)
        logger.info(f"  Extracted {len(tables)} tables")

        # Phase 5: Build section hierarchy
        sections = self._build_sections(all_blocks, header_thresholds)
        logger.info(f"  Built {len(sections)} sections")

        # Phase 6: Extract company name from first few pages
        company_name = self._extract_company_name(all_blocks, path.stem)

        doc.close()

        parsed_doc = ParsedDocument(
            filename=path.name,
            company_name=company_name,
            total_pages=total_pages,
            sections=sections,
            tables=tables,
            metadata={
                "file_path": str(path),
                "header_font_sizes": header_thresholds,
                "body_font_size": body_size,
            }
        )

        logger.info(f"  Company: {company_name}")
        logger.info(f"  Parsing complete: {len(sections)} sections, {len(tables)} tables")
        return parsed_doc

    def _extract_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """Extract text blocks with font metadata from all pages."""
        blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Get detailed text blocks with font info
            block_dicts = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in block_dicts.get("blocks", []):
                if block.get("type") != 0:  # Skip image blocks
                    continue

                for line in block.get("lines", []):
                    line_text = ""
                    max_font_size = 0
                    is_bold = False

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        line_text += span.get("text", "")
                        font_size = span.get("size", 0)
                        max_font_size = max(max_font_size, font_size)
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or "black" in font_name:
                            is_bold = True

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    # Skip page numbers, headers/footers (very short text at margins)
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    if len(line_text) < 5 and (bbox[1] < 50 or bbox[3] > 780):
                        continue

                    blocks.append(TextBlock(
                        text=line_text,
                        page_number=page_num + 1,  # 1-indexed
                        font_size=max_font_size,
                        is_bold=is_bold,
                        bbox=bbox,
                    ))

        return blocks

    def _classify_blocks(
        self,
        blocks: List[TextBlock],
        header_thresholds: List[float],
        body_size: float
    ):
        """Classify blocks as headers vs body text based on font size."""
        for block in blocks:
            text_clean = block.text.strip().lower()

            # Skip very short lines or lines that are just numbers
            if len(text_clean) < 3 or text_clean.replace(".", "").replace(",", "").isdigit():
                continue

            is_header = False

            # Font-size based detection
            if block.font_size > body_size + 1.0:
                is_header = True

            # Bold text matching known section patterns
            if block.is_bold and any(s in text_clean for s in self.KNOWN_SECTIONS):
                is_header = True

            # All-caps short text is often a header
            if block.text.isupper() and len(block.text.split()) <= 10:
                is_header = True

            # Numbered section headers like "1.", "1.1", "Section 1"
            if re.match(r'^(\d+\.?\d*\.?\s+|section\s+\d)', text_clean):
                if block.font_size >= body_size:
                    is_header = True

            if is_header:
                block.block_type = "header"

    def _extract_tables(self, doc: fitz.Document) -> List[TextBlock]:
        """Extract tables from the PDF using PyMuPDF's table detection."""
        tables = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                page_tables = page.find_tables()
                for table_idx, table in enumerate(page_tables):
                    # Extract table data as text
                    table_data = table.extract()
                    if not table_data:
                        continue

                    # Format as readable text
                    rows = []
                    for row in table_data:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_text = "\n".join(rows)

                    if table_text.strip():
                        tables.append(TextBlock(
                            text=table_text,
                            page_number=page_num + 1,
                            font_size=0,
                            is_bold=False,
                            bbox=table.bbox if hasattr(table, 'bbox') else (0, 0, 0, 0),
                            block_type="table",
                        ))
            except Exception as e:
                logger.debug(f"  Table extraction failed on page {page_num + 1}: {e}")
                continue

        return tables

    def _build_sections(
        self,
        blocks: List[TextBlock],
        header_thresholds: List[float]
    ) -> List[ParsedSection]:
        """Build hierarchical sections from classified blocks."""
        sections = []
        current_section = None
        current_body_lines = []

        for block in blocks:
            if block.block_type == "header":
                # Save previous section
                if current_section is not None:
                    current_section.raw_text = "\n".join(current_body_lines)
                    current_section.page_end = (
                        blocks[blocks.index(block) - 1].page_number
                        if blocks.index(block) > 0
                        else current_section.page_start
                    )
                    sections.append(current_section)

                # Determine header level based on font size
                level = 1
                if header_thresholds:
                    for i, threshold in enumerate(header_thresholds):
                        if block.font_size >= threshold - 0.5:
                            level = i + 1
                            break
                    else:
                        level = len(header_thresholds)

                current_section = ParsedSection(
                    title=block.text.strip(),
                    level=level,
                    page_start=block.page_number,
                    page_end=block.page_number,
                )
                current_body_lines = []
            else:
                current_body_lines.append(block.text)
                if current_section is not None:
                    current_section.blocks.append(block)

        # Save last section
        if current_section is not None:
            current_section.raw_text = "\n".join(current_body_lines)
            if blocks:
                current_section.page_end = blocks[-1].page_number
            sections.append(current_section)

        # If no sections were detected, create one big section
        if not sections and blocks:
            sections.append(ParsedSection(
                title="Document Content",
                level=1,
                page_start=1,
                page_end=blocks[-1].page_number,
                blocks=blocks,
                raw_text="\n".join(b.text for b in blocks),
            ))

        return sections

    def _extract_company_name(self, blocks: List[TextBlock], fallback: str) -> str:
        """Extract company name from the first page (usually largest text)."""
        first_page_blocks = [b for b in blocks if b.page_number <= 2]
        if not first_page_blocks:
            return fallback

        # The company name is typically the largest text on the first page
        largest = max(first_page_blocks, key=lambda b: b.font_size)
        name = largest.text.strip()

        # Clean up — remove "LIMITED", extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()

        # If the detected name is too long or too short, use filename
        if len(name) > 100 or len(name) < 3:
            return fallback

        return name
