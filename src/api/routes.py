from typing import Dict,Any,Optional
from src.retrieval.retriever import retrieve_context,is_context_relevant
from src.llm.llm_client import stream_llm_response

def query_pipeline(
    question: str,
    subject: str = "All Subjects",
    chat_history: str = "",
    engine: str = "⚡ Auto (Groq Fast ➡️ Ollama Backup)",
    local_model: str = "llama3.2:latest"
) -> Dict[str, Any]:
    """
    Execute end-to-end RAG query pipeline programmatically.
    Returns dictionary with question, subject, context, and generated answer.
    """
    context=retrieve_context(query=question,subject_filter=subject)
    is_relevant,fallback_msg=is_context_relevant(
        query=question,
        context=context,
        active_subject=subject
    )

    if not is_relevant:
        return {
            "question": question,
            "subject": subject,
            "context": context,
            "relevant": False,
            "answer": fallback_msg
        }

    full_answer=""
    for chunk in stream_llm_response(
        active_subject=subject,
        context=context,
        question=question,
        chat_history=chat_history,
        selected_engine=engine,
        local_model=local_model
    ):
        full_answer+=chunk

    return {
        "question": question,
        "subject": subject,
        "context": context,
        "relevant": True,
        "answer": full_answer
    }

def answer_query(question: str, subject: str = "All Subjects") -> str:
    """Convenience helper returning just the text response."""
    result=query_pipeline(question=question, subject=subject)
    return result.get("answer","")
