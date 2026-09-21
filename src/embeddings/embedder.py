import os
from typing import List,Union
import numpy as np

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"

try:
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from src.utils.helpers import load_app_config

_EMBEDDER_INSTANCE=None


class EmbeddingModel:
    """Wrapper class for generating text embeddings using SentenceTransformer."""

    def __init__(self,model_name: str ="all-MiniLM-L6-v2",device: str = "cpu"):
        self.model_name=model_name
        self.device=device
        self.model=None
        self._load_model()

    def _load_model(self):
        try:
            self.model=SentenceTransformer(self.model_name,device=self.device)
            print(f"Embedding model loaded: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"Error loading embedding model {self.model_name}: {e}")
            raise

    def encode(self,texts: Union[str,List[str]],show_progress_bar: bool=False) -> np.ndarray:
        """Generate embedding vector(s) for string or list of strings."""
        if not self.model:
            raise ValueError("Embedding model is not initialized.")
        if isinstance(texts,str):
            texts=[texts]
        embeddings=self.model.encode(texts,show_progress_bar=show_progress_bar)
        return embeddings

    def encode_single(self,text: str) -> List[float]:
        """Generate a single 1D embedding list of floats for a query string."""
        emb=self.model.encode(text,show_progress_bar=False)
        if isinstance(emb,np.ndarray):
            if emb.ndim==2:
                return emb[0].tolist()
            return emb.tolist()
        return list(emb)


def get_embedding_model(model_name: str=None) -> EmbeddingModel:
    """Singleton getter for the shared embedding model."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        cfg=load_app_config()
        m_name=model_name or cfg.get("embeddings",{}).get("model_name","all-MiniLM-L6-v2")
        device=cfg.get("embeddings",{}).get("device","cpu")
        _EMBEDDER_INSTANCE=EmbeddingModel(model_name=m_name,device=device)
    return _EMBEDDER_INSTANCE