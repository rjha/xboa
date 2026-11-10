import asyncio
import logging
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger("xboa.fastapi.main." + __name__)

class AppContext:
    """
    A unified in-memory object holding all long-lived 
    container components that are used by api endpoints 
    across the fastapi lifecycle process 
    """
    def __init__(self, api_keys: dict[str, str], db_uri: str):
        logger.info("Initializing container app context...")
        
        self.db_string: str = db_uri
        self.do_async_client: AsyncOpenAI = AsyncOpenAI(
            base_url="https://inference.do-ai.run/v1/",
            api_key=api_keys["DO_INFERENCE_KEY"],
            timeout=60.0,
        )

        self.do_blocking_client: OpenAI =  OpenAI(
            base_url="https://inference.do-ai.run/v1/", 
            api_key=api_keys["DO_INFERENCE_KEY"],
            timeout=60.0
        )
       
        logger.info("container app context initialized")

    def close(self):
        """release all resources cleanly."""
        logger.info("Releasing container app context resources")
        try:
            asyncio.run(self.do_async_client.close())
        except Exception as e:
            logger.error(f"Error closing do_openai_client connection pool: {e}")
