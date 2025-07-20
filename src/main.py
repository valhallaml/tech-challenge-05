import os
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse

from api.router import api_router
from core.configs import settings
from dotenv import load_dotenv

app = FastAPI(
    title = 'Datathon API',
    description = 'Datathon API',
    summary = '',
    version = '1.0.0'
)

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('src/favicon.ico')

@app.get('/', response_class=HTMLResponse)
async def home():
    with open('src/home.html') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == '__main__':
    load_dotenv()
    environment = os.getenv('ENVIRONMENT', 'development')
    is_dev = environment == 'development'
    uvicorn.run(app = 'main:app', host = '0.0.0.0', port = 8000, reload = is_dev)
