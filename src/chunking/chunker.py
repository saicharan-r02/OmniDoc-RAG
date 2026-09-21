import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.helpers import load_app_config


class DocumentChunker:
    """Handles recursive text splitting and document chunk metadata preparation."""

    def __init__(self,chunk_size: int =1000,chunk_overlap: int =200):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n","\n"," ",""]
        )

    def split_documents(self,docs: List[Document],root_data_dir: str = "./PDF_Data") -> List[Document]:
        """Split incoming documents into chunks while preserving and enriching metadata."""
        split_docs: List[Document]=[]
        for doc in docs:
            chunks=self.splitter.split_text(doc.page_content)
            for chunk in chunks:
                if not chunk.strip():
                    continue
                source_path=doc.metadata.get("source","")
                try:
                    rel=os.path.relpath(source_path,root_data_dir).replace("\\","/")
                    parts=[p for p in rel.split("/") if p and p!="."]
                    subject_name=parts[0] if len(parts)>1 else "General"
                except Exception:
                    subject_name="General"

                metadata={
                    **doc.metadata,
                    "source": source_path,
                    "subject": subject_name,
                    "content_length": len(chunk)
                }
                split_docs.append(Document(page_content=chunk,metadata=metadata))
        return split_docs


def split_documents(docs: List[Document],root_data_dir: str ="./PDF_Data") -> List[Document]:
    """Functional interface for chunking documents using configuration settings."""
    cfg=load_app_config()
    chunk_size=cfg.get("chunking", {}).get("chunk_size",1000)
    chunk_overlap=cfg.get("chunking", {}).get("chunk_overlap",200)
    chunker=DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_documents(docs, root_data_dir=root_data_dir)
