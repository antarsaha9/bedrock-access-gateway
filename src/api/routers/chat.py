import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from api.setting import DEFAULT_MODEL, ENABLE_RESPONSE_CACHE
from api.cache_worker import generate_cache_key, read_from_cache, write_to_cache

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@router.post("/completions", response_model=ChatResponse | ChatStreamResponse, response_model_exclude_unset=True)
async def chat_completions(
        chat_request: Annotated[
            ChatRequest,
            Body(
                examples=[
                    {
                        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"},
                        ],
                    }
                ],
            ),
        ]
):
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    if ENABLE_RESPONSE_CACHE and chat_request.cache and chat_request.invalidate_cache is not True:
        request_dict = chat_request.model_dump()
        cache_key = generate_cache_key(request_dict)

        # Check cache
        cached_result = read_from_cache(cache_key)
        if cached_result:
            logger.debug("Cache key: %s", cache_key)
            return ChatResponse.model_validate(cached_result)

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request), media_type="text/event-stream"
        )
    response = await run_in_threadpool(model.chat, chat_request)  # used run_in_threadpool for concurrent threads
    if ENABLE_RESPONSE_CACHE and chat_request.cache:
        # Store the response in the cache
        write_to_cache(cache_key, request_dict, response.model_dump())
    return response
