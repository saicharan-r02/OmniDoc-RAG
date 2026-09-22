import os
import gc
import json
import asyncio
from typing import Optional,AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.retrieval.retriever import retrieve_context, is_context_relevant
from src.llm.llm_client import stream_llm_response, ALL_GROQ_MODELS
from src.vectordb.vector_store import get_subject_counts
from src.utils.helpers import SUBJECT_METADATA,SIDEBAR_CATEGORIES,SUBJECT_SAMPLE_QUESTIONS,get_groq_api_key,normalize_subject_name

class ChatRequest(BaseModel):
    question: str =Field(..., min_length=1, max_length=4000)
    subject: str =Field(default="All Subjects")
    chat_history: str =Field(default="")
    engine: str =Field(default="Auto Cascading Pool")
    custom_api_key: Optional[str] =Field(default="")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memory-safe startup for low-RAM cloud instances."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
    except Exception:
        pass
    gc.collect()
    print("OmniDoc-RAG API ready.")
    yield

app=FastAPI(
    title="OmniDoc-RAG API",
    description="Academic RAG Assistant — Production FastAPI Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health",tags=["System"])
async def health_check():
    """Returns system status and basic statistics."""
    try:
        counts=get_subject_counts()
        total=sum(counts.values()) if counts else 0
    except Exception:
        total=0

    groq_key=get_groq_api_key()

    return {
        "status":"ok",
        "total_chunks":total,
        "subjects":len(SUBJECT_METADATA),
        "groq_configured":bool(groq_key),
        "available_models":ALL_GROQ_MODELS,
    }

@app.get("/api/subjects",tags=["Subjects"])
async def get_subjects():
    """Returns all subjects grouped by category."""
    categories={}
    for cat_label,subject_keys in SIDEBAR_CATEGORIES.items():
        subjects_in_cat=[]
        for key in subject_keys:
            meta=SUBJECT_METADATA.get(key,{})
            subjects_in_cat.append({
                "key":key,
                "title":meta.get("title",key),
                "icon":meta.get("icon","📚"),
                "type":meta.get("type","Notes"),
                "sample_questions":SUBJECT_SAMPLE_QUESTIONS.get(key,[]),
            })
        categories[cat_label]=subjects_in_cat
    return {"categories":categories}

@app.post("/api/chat",tags=["Chat"])
async def chat_stream(req: ChatRequest,request: Request):
    """
    Streams token-by-token AI responses via Server-Sent Events (SSE).
    Uses server's configured GROQ_API_KEY or header/custom key.
    """
    header_key=request.headers.get("x-groq-api-key","").strip()
    active_key=req.custom_api_key.strip() or header_key or get_groq_api_key()

    async def event_generator() -> AsyncGenerator[str,None]:
        loop=asyncio.get_event_loop()
        normalized_subj=normalize_subject_name(req.subject) or "All Subjects"

        yield ": connected\n\n"

        if not active_key:
            yield f"data: {json.dumps({'type':'token', 'content':'⚠️ **Groq API Key Required:** Please configure `GROQ_API_KEY` in your environment or Settings modal.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        retrieval_future=loop.run_in_executor(
            None,
            lambda: retrieve_context(query=req.question,subject_filter=normalized_subj)
        )
        try:
            while not retrieval_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(retrieval_future),timeout=5)
                except asyncio.TimeoutError:
                    yield ": retrieval-in-progress\n\n"
            context=await retrieval_future
        except Exception as exc:
            print(f"Retrieval failed: {exc}")
            yield f"data: {json.dumps({'type':'token','content':'⚠️ Document search is temporarily unavailable. Please verify the vector database configuration and try again.'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        is_relevant,fallback_msg=is_context_relevant(
            query=req.question,
            context=context,
            active_subject=normalized_subj
        )

        if not is_relevant:
            yield f"data: {json.dumps({'type':'token','content':fallback_msg})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        shifts = []

        def on_model_shift(current: str, next_model: str):
            shifts.append({"from":current,"to":next_model})

        def blocking_stream():
            if active_key and not os.getenv("GROQ_API_KEY"):
                os.environ["GROQ_API_KEY"] = active_key
            return list(stream_llm_response(
                active_subject=normalized_subj,
                context=context,
                question=req.question,
                chat_history=req.chat_history,
                selected_engine=req.engine,
                on_fallback=on_model_shift,
            ))

        generation_future=loop.run_in_executor(None,blocking_stream)
        try:
            while not generation_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(generation_future), timeout=5)
                except asyncio.TimeoutError:
                    yield ": generation-in-progress\n\n"
            chunks=await generation_future
            if not chunks:
                yield f"data: {json.dumps({'type':'token','content':'⚠️ The AI service returned no response. Please check the Groq API key and model availability.'})}\n\n"
            for chunk in chunks:
                if chunk:
                    yield f"data: {json.dumps({'type':'token','content':chunk})}\n\n"

            if shifts:
                yield f"data: {json.dumps({'type':'shift_notice','shifts':shifts})}\n\n"

            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'token','content': f'⚠️ AI Generation Notice: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no-cache",
            "X-Accel-Buffering":"no",
        }
    )

frontend_path=os.path.join(os.path.dirname(__file__),"frontend")
if os.path.exists(frontend_path):
    app.mount("/static",StaticFiles(directory=frontend_path),name="static")

    @app.get("/",include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path,"index.html"))


if __name__=="__main__":
    import uvicorn
    port=int(os.environ.get("PORT",10000))
    print(f"Starting OmniDoc-RAG on port {port}...")
    uvicorn.run(app, host="0.0.0.0",port=port,log_level="info")