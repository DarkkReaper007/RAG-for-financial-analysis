"""
Stage 2: Hierarchical Chunking

Implements the HiChunk-style parent-child chunking strategy.
Parent chunks (full sections) provide context for generation, while
child chunks (semantic sub-sections) are used for precise retrieval.
"""

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from src.ingestion.pdf_parser import ParsedDocument, ParsedSection, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text with metadata for retrieval."""
    chunk_id: str
    text: str
    chunk_type: str  # "parent", "child", "table"
    parent_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "parent_id": self.parent_id or "",
            **self.metadata,
        }


class HierarchicalChunker:
    """
    Hierarchical chunking engine implementing the HiChunk Auto-Merge pattern.
    
    Creates parent chunks (full sections) and child chunks (semantic sub-sections)
    with parent-child linking for context expansion during generation.
    
    Args:
        child_chunk_size: Target size for child chunks in characters.
        child_overlap: Overlap between adjacent child chunks in characters.
        min_chunk_size: Minimum viable chunk size (smaller chunks are merged).
    """

    def __init__(
        self,
        child_chunk_size: int = 1000,
        child_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """
        Chunk a parsed document into hierarchical parent-child chunks.
        
        Args:
            parsed_doc: A ParsedDocument from the PDF parser.
            
        Returns:
            List of Chunk objects (both parent and child chunks).
        """
        all_chunks = []
        logger.info(f"Chunking document: {parsed_doc.filename}")
        logger.info(f"  Sections to process: {len(parsed_doc.sections)}")

        # Process each section
        for section in parsed_doc.sections:
            section_text = section.get_full_text().strip()
            if len(section_text) < self.min_chunk_size:
                continue

            # Create parent chunk (full section)
            parent_id = str(uuid.uuid4())
            parent_chunk = Chunk(
                chunk_id=parent_id,
                text=section_text,
                chunk_type="parent",
                metadata={
                    "company_name": parsed_doc.company_name,
                    "filename": parsed_doc.filename,
                    "section_title": section.title,
                    "section_level": section.level,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                }
            )
            all_chunks.append(parent_chunk)

            # Create child chunks from the section
            children = self._create_child_chunks(
                section_text, parent_id, section, parsed_doc
            )
            all_chunks.extend(children)

        # Process tables as separate chunks
        for table in parsed_doc.tables:
            if len(table.text.strip()) < self.min_chunk_size:
                continue

            table_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=table.text,
                chunk_type="table",
                metadata={
                    "company_name": parsed_doc.company_name,
                    "filename": parsed_doc.filename,
                    "section_title": "Financial Table",
                    "page_start": table.page_number,
                    "page_end": table.page_number,
                }
            )
            all_chunks.append(table_chunk)

        parent_count = sum(1 for c in all_chunks if c.chunk_type == "parent")
        child_count = sum(1 for c in all_chunks if c.chunk_type == "child")
        table_count = sum(1 for c in all_chunks if c.chunk_type == "table")

        logger.info(f"  Created {len(all_chunks)} chunks: "
                     f"{parent_count} parent, {child_count} child, {table_count} table")

        return all_chunks

    def _create_child_chunks(
        self,
        text: str,
        parent_id: str,
        section: ParsedSection,
        parsed_doc: ParsedDocument,
    ) -> List[Chunk]:
        """
        Split section text into semantic child chunks.
        
        First attempts to split on paragraph/topic boundaries.
        Falls back to character-based splitting with overlap.
        """
        children = []

        # Try semantic splitting first (paragraph boundaries)
        paragraphs = self._split_into_paragraphs(text)

        if len(paragraphs) > 1:
            # Merge small paragraphs together to reach target chunk size
            children = self._merge_paragraphs_into_chunks(
                paragraphs, parent_id, section, parsed_doc
            )
        else:
            # Fall back to character-based splitting with overlap
            children = self._split_by_characters(
                text, parent_id, section, parsed_doc
            )

        return children

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on double newlines or topic transitions."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)

        # Also split on numbered list items if they're substantial
        refined_paragraphs = []
        for para in paragraphs:
            # Check if paragraph contains numbered sub-sections
            sub_sections = re.split(r'\n(?=\d+\.\s)', para)
            if len(sub_sections) > 1 and all(len(s.strip()) > 50 for s in sub_sections):
                refined_paragraphs.extend(sub_sections)
            else:
                refined_paragraphs.append(para)

        return [p.strip() for p in refined_paragraphs if p.strip()]

    def _merge_paragraphs_into_chunks(
        self,
        paragraphs: List[str],
        parent_id: str,
        section: ParsedSection,
        parsed_doc: ParsedDocument,
    ) -> List[Chunk]:
        """Merge small paragraphs into chunks of target size."""
        chunks = []
        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, save current and start new
            if (current_text and
                    len(current_text) + len(para) > self.child_chunk_size):
                chunks.append(self._make_child_chunk(
                    current_text, parent_id, section, parsed_doc, chunk_index
                ))
                chunk_index += 1
                # Keep overlap from end of current chunk
                overlap_text = current_text[-self.child_overlap:] if self.child_overlap > 0 else ""
                current_text = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_text = current_text + "\n\n" + para if current_text else para

        # Save last chunk
        if current_text.strip() and len(current_text.strip()) >= self.min_chunk_size:
            chunks.append(self._make_child_chunk(
                current_text, parent_id, section, parsed_doc, chunk_index
            ))

        return chunks

    def _split_by_characters(
        self,
        text: str,
        parent_id: str,
        section: ParsedSection,
        parsed_doc: ParsedDocument,
    ) -> List[Chunk]:
        """Split text into chunks by character count with overlap."""
        chunks = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = start + self.child_chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for the last sentence-ending punctuation before the limit
                last_period = text.rfind('. ', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                if break_point > start + self.min_chunk_size:
                    end = break_point + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._make_child_chunk(
                    chunk_text, parent_id, section, parsed_doc, chunk_index
                ))
                chunk_index += 1

            start = end - self.child_overlap

        return chunks

    def _make_child_chunk(
        self,
        text: str,
        parent_id: str,
        section: ParsedSection,
        parsed_doc: ParsedDocument,
        chunk_index: int,
    ) -> Chunk:
        """Create a child chunk with metadata."""
        # Estimate page number based on position in section
        total_section_len = len(section.get_full_text())
        if total_section_len > 0:
            position_ratio = min(1.0, chunk_index * self.child_chunk_size / total_section_len)
        else:
            position_ratio = 0
        estimated_page = section.page_start + int(
            (section.page_end - section.page_start) * position_ratio
        )

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            chunk_type="child",
            parent_id=parent_id,
            metadata={
                "company_name": parsed_doc.company_name,
                "filename": parsed_doc.filename,
                "section_title": section.title,
                "section_level": section.level,
                "page_start": estimated_page,
                "page_end": estimated_page,
                "chunk_index": chunk_index,
            }
        )
