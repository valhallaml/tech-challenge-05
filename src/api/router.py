from fastapi import APIRouter
from api.v1.datathon import router

api_router = APIRouter()
api_router.include_router(router, prefix='', tags=[ 'datathon' ])
