import logging 
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import APIRouter

# XBOA SDK
from softmaxx.config import AppConfig
from softmaxx.config import (
    get_logger_config,
    get_database_config
)
from softmaxx.api.service import register_exception_handlers
from softmaxx.api.context import AppContext
# LOCAL ROUTERS
from .routers.llm import llm_router
from .routers.game import game_router
from .routers.hello import hello_router


# noinspection PyUnusedLocal
@asynccontextmanager
async def lifespan(boa_app: FastAPI):
    logger = logging.getLogger("xboa.fastapi.main." + __name__)
    AppConfig.load()
    log_config = get_logger_config("fastapi")
    log_file_name = log_config.log_file
    AppConfig.init_logging(log_file=log_file_name, log_level=log_config.log_level)
    logger.info(f"fastapi main app started...")

    # Initialize the container context
    api_keys = AppConfig.get("api_keys")
    db_config = get_database_config()
    ctx = AppContext(api_keys=api_keys, db_uri=db_config.get_uri())
    boa_app.state.ctx = ctx

    yield
    # shutdown events
    logger.info("fastapi lifecycle components shutdown ...")
    ctx.close()
    logger.info("fastapi container is stopped.")


app = FastAPI(lifespan=lifespan)

# Register global exception handlers
register_exception_handlers(app)

main_router = APIRouter(prefix="/boa/v1")
main_router.include_router(hello_router)
main_router.include_router(llm_router)
main_router.include_router(game_router)
app.include_router(main_router)