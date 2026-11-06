from typing import AsyncGenerator, Optional
import json
import time 
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from softmaxx.config import AppConfig


"""
To forward the port on SSL for testing using VPC bound model key,
(1) add an entry in /etc/hosts to route inference.do-ai.run to 127.0.0.1 

# Do inference testing
# 127.0.0.1 inference.do-ai.run

(2) open a port to remote machine,
sudo ssh -i ~/.ssh/id_rsa -L 443:inference.do-ai.run:443 user@remote-ip

"""


# do not include a final slash in the prefix

llm_router = APIRouter(
    prefix="/llm",
    tags=["LLM"],
    responses={404: {"description": "LLM router not found"}},
)


# Global holder for client instance
_client: Optional[AsyncOpenAI] = None


def get_llm_client() -> AsyncOpenAI:
    """Dependency that lazily initializes and returns the AsyncOpenAI client."""
    global _client
    if _client is None:
        api_keys = AppConfig.get("api_keys")
        _client = AsyncOpenAI(
            base_url="https://inference.do-ai.run/v1/",
            api_key=api_keys["DO_SONNET_055"],
            timeout=60.0,
        )
    return _client


async def event_generator(prompt: str, client: AsyncOpenAI) -> AsyncGenerator[str, None]:
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    print(f"\n[LLM Request] Started prompt: '{prompt[:30]}...'")

    try:
        response = await client.chat.completions.create(
            model="anthropic-claude-sonnet-5.5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            stream=True
        )

        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Capture standard content OR reasoning/thinking content
                text_chunk = getattr(delta, "content", None) or getattr(
                    delta, "reasoning_content", None
                )

                if text_chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - start_time) * 1000
                        print(f"[LLM Timing] ⚡ TTFT: {ttft_ms:.2f} ms")

                    token_count += 1
                    yield f"data: {json.dumps({'token': text_chunk})}\n\n"

        total_time_ms = (time.perf_counter() - start_time) * 1000
        print(
            f"[LLM Timing] ✅ Stream complete. Total Time: {total_time_ms:.2f} ms | Chunks: {token_count}"
        )

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_time_ms = (time.perf_counter() - start_time) * 1000
        print(f"[LLM Timing] ❌ Error after {error_time_ms:.2f} ms: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@llm_router.post("/qwen3/chat")
async def process_qwen3_chat(
    prompt: str = "Explain the Chaos theory.",
    client: AsyncOpenAI = Depends(get_llm_client)):

    """Streams JSON SSE events live as tokens arrive."""
    return StreamingResponse(
        event_generator(prompt, client),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables buffering in Nginx automatically
        },
    )