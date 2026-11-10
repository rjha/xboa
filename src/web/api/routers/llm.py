import json
import time 
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from collections.abc import AsyncGenerator
from pydantic import BaseModel
from openai import AsyncOpenAI
import softmaxx.boxgpt.rag as box_rag
import softmaxx.boxgpt.prompt as box_prompt
from softmaxx.boxgpt.encoder import OpenAIBGEM3Encoder


logger = logging.getLogger("xboa.fastapi.main." + __name__)

# do not include a final slash in the prefix
llm_router = APIRouter(
    prefix="/llm",
    tags=["LLM"],
    responses={404: {"description": "LLM router not found"}},
)

class ChatRequest(BaseModel):
    prompt: str
    model: str
    max_tokens: int

class EmbeddingRequest(BaseModel):
    text: list[str]
    model: str

class SearchRequest(BaseModel):
    user_query: str
    model: str
    max_tokens: int


async def event_generator(
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        client: AsyncOpenAI
) -> AsyncGenerator[str, None]:
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    # messages is [{"role": "user", "content": prompt}]
    user_query = messages[-1].get("content", "")[:80].replace("\n", " ")
    logger.info(f"\n[LLM Query] {user_query}")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                text_chunk = getattr(delta, "content", None) or getattr(
                    delta, "reasoning_content", None
                )

                if text_chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - start_time) * 1000
                        logger.info(f"[LLM Time to first token: {ttft_ms:.2f} ms")

                    token_count += 1
                    yield f"data: {json.dumps({'token': text_chunk})}\n\n"

        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "[Success] Stream complete "
            f"Total Time: {total_time_ms:.2f} ms | "
            f"Chunks: {token_count}"
        )

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_time_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Error after {error_time_ms:.2f} ms: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"



@llm_router.post("/chat/complete")
async def process_chat_complete(request: Request, body: ChatRequest):
    """
    Streams JSON SSE events live as tokens arrive 
    using the requested model name.
    """
    client = request.app.state.ctx.do_async_client
    messages = [{"role": "user", "content": body.prompt}]

    return StreamingResponse(
        event_generator(
            messages=messages, 
            model=body.model, 
            max_tokens=body.max_tokens,
            client=client
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@llm_router.post("/embedding/create")
async def create_embedding(request: Request, body: EmbeddingRequest):
    """
    Generates text embeddings using the DO openai client 
    in the context and return the api response directly.
    """
    client = request.app.state.ctx.do_blocking_client 
    response = client.embeddings.create(
        input=body.text,
        model=body.model,
        encoding_format="float"


    )
    
    return response


@llm_router.post("/doc/search")
async def search_and_generate(request: Request, body: SearchRequest):
    """
    1. Generates text query embeddings using do_blocking_client.
    2. Search postgres  for matching vectors.
    3. format prompt payoads with search result vectors
    4. Streams token arrays using do_async_client
    """
    ctx = request.app.state.ctx
    
    # Instantiate online text encoder using the blocking client
    encoder = OpenAIBGEM3Encoder(api_client=ctx.do_blocking_client)
    mmr_chunks = box_rag.search_doc_chunks(
        db_string=ctx.db_string,
        text_encoder=encoder,
        user_query=body.user_query
    )
    
    llm_messages = box_prompt.rag_query_message(body.user_query, mmr_chunks)
    return StreamingResponse(
        event_generator(
            messages=llm_messages, 
            model=body.model, 
            max_tokens=body.max_tokens, 
            client=ctx.do_async_client
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
