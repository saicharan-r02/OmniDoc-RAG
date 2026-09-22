import socket
from typing import List,Generator,Any,Optional,Callable
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
try:
    from langchain_groq import ChatGroq
    from groq import Groq
except ImportError:
    ChatGroq=None
    Groq=None

from src.prompts.prompt_templates import PROMPT_TEMPLATE
from src.utils.helpers import load_app_config,get_groq_api_key

ALL_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "groq/compound",
    "allam-2-7b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]


def is_ollama_online(host: str="127.0.0.1",port: int=11434,timeout: float=0.5) -> bool:
    """Quick socket check to verify if the local Ollama server is running."""
    try:
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result=sock.connect_ex((host,port))
        sock.close()
        return result==0
    except Exception:
        return False

def get_available_groq_models(api_key: Optional[str] =None) -> List[str]:
    """Fetch all available chat models from the user's Groq account."""
    key=api_key or get_groq_api_key()
    if not key or not Groq:
        return ALL_GROQ_MODELS

    try:
        client=Groq(api_key=key)
        all_models=client.models.list()
        chat_models=[]
        for m in all_models.data:
            mid=m.id
            if "whisper" not in mid and "guard" not in mid and "orpheus" not in mid:
                chat_models.append(mid)
        if chat_models:
            return chat_models
    except Exception:
        pass

    return []


def _is_model_unavailable_error(error: Exception) -> bool:
    """Return whether Groq rejected a model because it is missing or inaccessible."""
    message=str(error).lower()
    return "404" in message or "model_not_found" in message or "does not exist" in message


def _is_rate_limit_error(error: Exception) -> bool:
    """Return whether Groq asked the client to retry later."""
    message=str(error).lower()
    return "429" in message or "rate limit" in message or "rate_limit" in message


def get_local_ollama_models() -> List[str]:
    """Fetch locally installed Ollama models only if the Ollama daemon is active."""
    if not is_ollama_online():
        return []
    try:
        models_list=ollama.list().get("models",[])
        names=[m.get("name") or m.get("model") for m in models_list if m]
        if names:
            return names
    except Exception:
        pass
    return ["llama3.2:latest","qwen2.5:7b","llama3.1:8b","mistral:7b"]

def create_groq_client(model_name: str="openai/gpt-oss-120b",api_key: Optional[str]=None,temperature: float=0.0) -> Optional[Any]:
    """Create a LangChain ChatGroq instance."""
    if not ChatGroq:
        return None
    key=api_key or get_groq_api_key()
    if not key:
        return None
    return ChatGroq(model=model_name,groq_api_key=key,temperature=temperature)


def create_ollama_client(model_name: str = "llama3.2:latest",temperature: float = 0.0) -> Optional[ChatOllama]:
    """Create a LangChain ChatOllama instance if server is online."""
    if not is_ollama_online():
        return None
    return ChatOllama(model=model_name,temperature=temperature)


def _extract_selected_groq_model(selected_engine: str) -> Optional[str]:
    """Extract model ID if explicitly chosen in the dropdown."""
    for m in ALL_GROQ_MODELS:
        if m in selected_engine:
            return m
    return None


def stream_llm_response(
    active_subject: str,
    context: str,
    question: str,
    chat_history: str = "",
    selected_engine: str = "⚡ Auto Cascading Pool (Rotates on Limit)",
    local_model: str = "llama3.2:latest",
    on_fallback: Optional[Callable[[str, str], None]] = None
) -> Generator[str, None, None]:
    """
    Unified streaming generator with multi-model cascading fallback.
    When a Groq model completes its limit (RateLimit, 429, TPM/RPM cap) or 404,
    it automatically shifts to the next model in the pool until success.
    """
    cfg=load_app_config()
    temp=cfg.get("llm", {}).get("temperature", 0.0)
    prompt=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    input_payload = {"active_subject": active_subject,"context": context,"chat_history": chat_history,"question": question}

    groq_api_key=get_groq_api_key()
    last_groq_error=""
    unavailable_models=[]
    rate_limited_models=[]

    is_pure_ollama=("Ollama" in selected_engine) and ("Auto" not in selected_engine)

    if not is_pure_ollama:
        if not groq_api_key:
            yield (
                "⚠️ **Groq API Key Required**\n\n"
                "To enable cloud AI inference:\n"
                "1. On **Streamlit Cloud / Spaces**: Add `GROQ_API_KEY = \"gsk_...\"` in Secrets / Environment Variables.\n"
                "2. Locally: Add `GROQ_API_KEY=gsk_...` in `.env` file, or enter it in the sidebar."
            )
            return

        if not ChatGroq:
            yield "⚠️ **Error:** `langchain-groq` package is not installed."
            return

        explicit_model=_extract_selected_groq_model(selected_engine)
        discovered_models=get_available_groq_models(groq_api_key)
        if discovered_models:
            preferred_models=[m for m in ALL_GROQ_MODELS if m in discovered_models]
            additional_models=[m for m in discovered_models if m not in preferred_models]
            model_queue=preferred_models+additional_models
        else:
            model_queue = ["openai/gpt-oss-120b","openai/gpt-oss-20b","llama-3.3-70b-versatile"]

        if explicit_model and explicit_model in model_queue:
            model_queue.remove(explicit_model)
            model_queue.insert(0,explicit_model)

        for idx, g_model in enumerate(model_queue):
            try:
                llm_groq=ChatGroq(
                    model=g_model,
                    groq_api_key=groq_api_key,
                    temperature=temp,
                    max_tokens=8192,
                    max_retries=1
                )
                chain_groq=prompt | llm_groq

                chunk_received=False
                for chunk in chain_groq.stream(input_payload):
                    piece=chunk.content if hasattr(chunk,"content") else str(chunk)
                    if piece:
                        chunk_received = True
                        yield piece

                if chunk_received:
                    return

            except Exception as groq_err:
                last_groq_error = str(groq_err)
                if _is_model_unavailable_error(groq_err):
                    unavailable_models.append(g_model)
                elif _is_rate_limit_error(groq_err):
                    rate_limited_models.append(g_model)
                next_model=model_queue[idx+1] if idx+1<len(model_queue) else "Local Ollama"

                if on_fallback:
                    try:
                        on_fallback(g_model,next_model)
                    except Exception:
                        pass

                continue

    if is_ollama_online():
        try:
            llm_ollama=ChatOllama(model=local_model,temperature=temp)
            chain_ollama=prompt|llm_ollama

            for chunk in chain_ollama.stream(input_payload):
                piece=chunk.content if hasattr(chunk,"content") else str(chunk)
                yield piece
            return
        except Exception as ollama_err:
            yield f"⚠️ **Local Ollama Error:** {ollama_err}\n\nPlease ensure model `{local_model}` is pulled (`ollama pull {local_model}`)."
            return

    if is_pure_ollama:
        yield (
            "⚠️ **Local Ollama Not Running**\n\n"
            "Could not connect to Ollama at `http://localhost:11434`.\n\n"
            "- If running locally: Start Ollama with `ollama serve` in a terminal.\n"
            "- If in the cloud: Select **⚡ Auto Cascading Pool** in the sidebar."
        )
        return

    if rate_limited_models and not unavailable_models:
        yield (
            f"⚠️ **Groq Rate Limit Reached**\n\n`{last_groq_error}`\n\n"
            "All currently available Groq models are rate-limited. Please wait and try again."
        )
    elif unavailable_models and not rate_limited_models:
        yield (
            "⚠️ **No Compatible Groq Model Available**\n\n"
            "The configured model list contains models that are unavailable for this Groq account. "
            "Refresh the model list or select a currently available Groq model."
        )
    elif last_groq_error:
        yield f"⚠️ **Groq Generation Failed**\n\n`{last_groq_error}`"
    else:
        yield "⚠️ **No AI Engine Available:** Please check your `GROQ_API_KEY` or start local Ollama."