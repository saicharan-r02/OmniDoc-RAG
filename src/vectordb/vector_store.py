import os
import uuid
import zipfile
import re
import shutil
from typing import List,Set,Dict,Any,Optional
import numpy as np
from langchain_core.documents import Document
from src.utils.helpers import load_app_config

_VECTOR_STORE_INSTANCE=None


def ensure_vector_db_ready(persist_dir: str="./pdf_db/chromadb",zip_path: str = "./pdf_db.zip") -> bool:
    """
    Ensure the vector database exists locally. If absent, attempts extraction from local zip
    or download from cloud URL (for Streamlit Cloud and HuggingFace Spaces deployments).
    """
    sqlite_path = os.path.join(persist_dir,"chroma.sqlite3")
    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 0:
        return True

    # Attempt 1: Extract from local zip if present (case-insensitive search for Linux)
    candidate_zips=[
        zip_path,
        "./pdf_db.zip",
        "./PDF_db.zip",
        "./PDF_DB.zip",
        "pdf_db.zip",
        "PDF_db.zip",
        "PDF_DB.zip",
        "../pdf_db.zip",
        "../PDF_db.zip"
    ]

    for czip in candidate_zips:
        if os.path.exists(czip) and os.path.getsize(czip)>1000:
            print(f"📦 Unzipping local database package from {czip}...")
            try:
                with zipfile.ZipFile(czip,"r") as zf:
                    zf.extractall(".")
                if not os.path.exists(persist_dir) and os.path.exists("./chromadb"):
                    os.makedirs("./pdf_db",exist_ok=True)
                    shutil.move("./chromadb","./pdf_db/chromadb")
                if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path)>0:
                    print("✅ Vector database extracted successfully.")
                    return True
            except Exception as e:
                print(f"Extraction error for {czip}: {e}")

    # Attempt 2: Download from VECTOR_DB_URL if defined (Streamlit Cloud secret or env)
    cloud_url=os.getenv("VECTOR_DB_URL","")
    try:
        import streamlit as st
        if not cloud_url and "VECTOR_DB_URL" in st.secrets:
            cloud_url=st.secrets["VECTOR_DB_URL"]
    except Exception:
        pass

    if cloud_url:
        print("🌐 Downloading vector database package from cloud storage...")
        target_zip="./pdf_db.zip"
        try:
            if "drive.google.com" in cloud_url:
                match=re.search(r"/d/([a-zA-Z0-9_-]+)",cloud_url) or re.search(r"id=([a-zA-Z0-9_-]+)",cloud_url)
                file_id=match.group(1) if match else None

                downloaded=False
                try:
                    import gdown
                    if file_id:
                        gdown.download(id=file_id,output=target_zip,quiet=False)
                    else:
                        gdown.download(cloud_url,target_zip,quiet=False)
                    if os.path.exists(target_zip) and os.path.getsize(target_zip)>1000:
                        downloaded=True
                except Exception:
                    pass

                if not downloaded and file_id:
                    import requests
                    direct_url=f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    resp=requests.get(direct_url,stream=True,timeout=180)
                    with open(target_zip,"wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                    if os.path.exists(target_zip) and os.path.getsize(target_zip)>1000:
                        downloaded=True
            else:
                import requests
                resp=requests.get(cloud_url,stream=True,timeout=180)
                with open(target_zip, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)

            if os.path.exists(target_zip) and os.path.getsize(target_zip)>1000:
                with zipfile.ZipFile(target_zip, "r") as zf:
                    zf.extractall(".")
                if not os.path.exists(persist_dir) and os.path.exists("./chromadb"):
                    os.makedirs("./pdf_db", exist_ok=True)
                    shutil.move("./chromadb", "./pdf_db/chromadb")
                if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path)>0:
                    print("✅ Cloud database downloaded and extracted successfully.")
                    return True
        except Exception as e:
            print(f"Cloud download failed: {e}")

    return False


class VectorStoreManager:
    """Manages ChromaDB persistent client, collections, queries, and document additions."""

    def __init__(self, persist_dir: str="./pdf_db/chromadb",collection_name: str = "Document__C"):
        self.persist_dir=persist_dir
        self.collection_name=collection_name
        self.client=None
        self.collection=None
        self._initialize()

    def _initialize(self):
        try:
            ensure_vector_db_ready(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
            import chromadb
            from chromadb.config import Settings
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True
            )
            self.client=chromadb.PersistentClient(path=self.persist_dir,settings=settings)
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF embedding document for RAG"}
            )
            count = self.collection.count()
            print(f"ChromaDB ready: collection='{self.collection_name}', items={count}")
        except Exception as e:
            print(f"Warning: ChromaDB initialization at {self.persist_dir}: {e}")
            self.client=None
            self.collection=None

    def get_indexed_sources(self)->Set[str]:
        """Fetch all unique source file paths currently stored in the collection."""
        if not self.collection:
            return set()
        try:
            indexed_sources=set()
            offset=0
            page_size=1000
            while True:
                results = self.collection.get(
                    include=["metadatas"],
                    limit=page_size,
                    offset=offset
                )
                metadatas=results.get("metadatas",[])
                indexed_sources.update(
                    m.get("source") for m in metadatas if m and "source" in m
                )
                if len(metadatas)<page_size:
                    break
                offset+=page_size
            return indexed_sources
        except Exception as e:
            print(f"Warning: Could not fetch indexed sources: {e}")
            return set()

    def delete_records(self,subject: Optional[str] = None) -> int:
        """Delete indexed records only; source documents on disk are never touched."""
        if not self.collection:
            return 0
        try:
            deleted=0
            offset=0
            page_size=1000
            while True:
                filters={"subject": subject} if subject else None
                results=self.collection.get(
                    where=filters,
                    include=[],
                    limit=page_size,
                    offset=offset
                )
                ids=results.get("ids",[])
                if not ids:
                    break
                self.collection.delete(ids=ids)
                deleted+=len(ids)
                if len(ids)<page_size:
                    break
            return deleted
        except Exception as e:
            raise RuntimeError(f"Could not delete vector records safely: {e}") from e

    def add_documents(self,documents: List[Document],embeddings: np.ndarray,batch_size: int=500):
        """Add documents and embeddings in batches."""
        if not self.collection:
            raise RuntimeError("ChromaDB collection is not initialized.")
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if not documents:
            return

        for start in range(0,len(documents),batch_size):
            end=min(start+batch_size,len(documents))
            batch_docs=documents[start:end]
            batch_embeddings=embeddings[start:end]
            ids,metadatas,doc_texts,embedding_list=[],[],[],[]

            for i,(doc,embed) in enumerate(zip(batch_docs, batch_embeddings), start=start):
                doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)
                meta=dict(doc.metadata or {})
                meta["content_length"] = len(doc.page_content)
                metadatas.append(meta)
                doc_texts.append(doc.page_content)
                embedding_list.append(embed.tolist() if hasattr(embed, "tolist") else list(embed))

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embedding_list,
                    metadatas=metadatas,
                    documents=doc_texts
                )
            except Exception as e:
                print(f"Error adding batch to vector store: {e}")

    def query_similarity(self,query_embedding: Any,n_results: int = 8,where: Optional[Dict[str, Any]]=None,where_document: Optional[Dict[str,Any]]=None) -> List[str]:
        """Query documents by vector similarity with dimensional safety."""
        if not self.collection:
            return []
        try:
            if isinstance(query_embedding,np.ndarray):
                query_embedding=query_embedding.tolist()

            if isinstance(query_embedding,list):
                if len(query_embedding)>0 and isinstance(query_embedding[0],list):
                    if len(query_embedding[0])>0 and isinstance(query_embedding[0][0],list):
                        formatted_embeddings=query_embedding[0]
                    else:
                        formatted_embeddings=query_embedding
                else:
                    formatted_embeddings=[query_embedding]
            else:
                formatted_embeddings=[[float(x) for x in query_embedding]]

            kwargs={
                "query_embeddings":formatted_embeddings,
                "n_results": n_results
            }
            if where:
                kwargs["where"]=where
            if where_document:
                kwargs["where_document"]=where_document

            results=self.collection.query(**kwargs)
            docs=results.get("documents",[[]])[0]
            return docs
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []

    def keyword_search(self,keyword: str,limit: int=8,where: Optional[Dict[str,Any]] = None,) -> List[str]:
        """Find indexed documents containing a keyword without loading an embedding model."""
        if not self.collection or not keyword.strip():
            return []
        try:
            results=self.collection.get(
                where=where,
                where_document={"$contains": keyword.strip()},
                include=["documents"],
                limit=limit
            )
            return [doc for doc in results.get("documents",[]) if doc]
        except Exception as e:
            print(f"Error searching vector store by keyword: {e}")
            return []


def get_vector_store() -> Optional[VectorStoreManager]:
    """Singleton getter for VectorStoreManager with safe lazy initialization."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        try:
            cfg=load_app_config()
            p_dir=cfg.get("database",{}).get("persist_directory","./pdf_db/chromadb")
            c_name=cfg.get("database",{}).get("collection_name","Document__C")
            _VECTOR_STORE_INSTANCE=VectorStoreManager(persist_dir=p_dir, collection_name=c_name)
        except Exception as e:
            print(f"Failed to initialize VectorStoreManager: {e}")
            return None
    return _VECTOR_STORE_INSTANCE


def get_subject_counts() -> Dict[str, int]:
    """Return count of document chunks indexed (fast O(1) total count)."""
    try:
        manager=get_vector_store()
        if not manager or not manager.collection:
            return {}
        total = manager.collection.count()
        return {"Total": total} if total>0 else {}
    except Exception:
        return {}