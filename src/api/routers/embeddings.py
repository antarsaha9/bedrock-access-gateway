import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends

from api.auth import api_key_auth
from api.models.bedrock import get_embeddings_model
from api.schema import EmbeddingsRequest, EmbeddingsResponse
from api.setting import DEFAULT_EMBEDDING_MODEL, ENABLE_RESPONSE_CACHE
from api.cache_worker import generate_cache_key, read_from_cache, write_to_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter(
    prefix="/embeddings",
    dependencies=[Depends(api_key_auth)],
)


@router.post("", response_model=EmbeddingsResponse)
async def embeddings(
    embeddings_request: Annotated[
        EmbeddingsRequest,
        Body(
            examples=[
                {
                    "model": "cohere.embed-multilingual-v3",
                    "input": ["Your text string goes here"],
                }
            ],
        ),
    ],
):
    if embeddings_request.model.lower().startswith("text-embedding-"):
        embeddings_request.model = DEFAULT_EMBEDDING_MODEL

    if ENABLE_RESPONSE_CACHE and embeddings_request.cache and embeddings_request.invalidate_cache is not True:
        request_dict = embeddings_request.model_dump()
        cache_key = generate_cache_key(request_dict)

        # Check cache
        cached_result = read_from_cache(cache_key)
        if cached_result:
            logger.debug("Cache key: %s", cache_key)
            return EmbeddingsResponse.model_validate(cached_result)
    # Exception will be raised if model not supported.
    model = get_embeddings_model(embeddings_request.model)
    response = model.embed(embeddings_request)

    if ENABLE_RESPONSE_CACHE and embeddings_request.cache:
        # Store the response in the cache
        write_to_cache(cache_key, request_dict, response.model_dump())
    return response
