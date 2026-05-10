# frontend
# html페이지 띄워주는 라우터

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["pages"])

templates = Jinja2Templates(directory="app/templates")


@router.get("/mobile", response_class=HTMLResponse)
def mobile_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="mobile.html",
    )


@router.get("/pc", response_class=HTMLResponse)
def pc_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pc.html",
    )