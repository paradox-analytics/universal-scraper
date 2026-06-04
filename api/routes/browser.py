"""
Browser session routes — live preview with interactive browser control.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Form
from pydantic import BaseModel

from api.middleware.auth import get_tenant_id
from api.browser_session import get_session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/browser", tags=["browser"])


class BrowserSessionRequest(BaseModel):
    proxy_config: Optional[Dict[str, Any]] = None
    headless: bool = True
    viewport: Optional[Dict[str, int]] = None


class NavigateRequest(BaseModel):
    url: str
    wait_for: str = "domcontentloaded"
    timeout: int = 60000


class ClickRequest(BaseModel):
    selector: str
    button: str = "left"


class ScrollRequest(BaseModel):
    direction: str = "down"
    amount: int = 500


class SelectElementRequest(BaseModel):
    selector: str
    field_name: str


@router.post("/session")
async def create_browser_session(
    request: BrowserSessionRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        from api.main import convert_proxy_config
        proxy_config = convert_proxy_config(request.proxy_config) if request.proxy_config else None
        session = await manager.create_session(
            tenant_id=tenant_id,
            proxy_config=proxy_config,
            headless=request.headless,
            viewport=request.viewport,
        )
        return {"success": True, "session_id": session.id, "message": "Browser session created"}
    except Exception as e:
        logger.error(f"Failed to create browser session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def close_browser_session(
    session_id: str, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        success = await manager.close_session(session_id)
        return {"success": success, "message": "Session closed" if success else "Session not found"}
    except Exception as e:
        logger.error(f"Failed to close browser session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/navigate")
async def browser_navigate(
    session_id: str, request: NavigateRequest, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        return await manager.navigate(session_id=session_id, url=request.url, wait_for=request.wait_for, timeout=request.timeout)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Navigation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/click")
async def browser_click(
    session_id: str, request: ClickRequest, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        return await manager.click(session_id=session_id, selector=request.selector, button=request.button)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Click failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/scroll")
async def browser_scroll(
    session_id: str, request: ScrollRequest, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        return await manager.scroll(session_id=session_id, direction=request.direction, amount=request.amount)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Scroll failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/screenshot")
async def browser_screenshot(session_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = await get_session_manager()
        screenshot = await manager.get_screenshot(session_id)
        if screenshot:
            return {"success": True, "screenshot": screenshot}
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Screenshot failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/html")
async def browser_html(session_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = await get_session_manager()
        html = await manager.get_html(session_id)
        if html:
            return {"success": True, "html": html, "size": len(html)}
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HTML fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/select")
async def browser_select_element(
    session_id: str, request: SelectElementRequest, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        return await manager.select_element(session_id=session_id, selector=request.selector, field_name=request.field_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Element selection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/selections")
async def browser_get_selections(session_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = await get_session_manager()
        selections = await manager.get_selected_elements(session_id)
        return {"success": True, "selections": selections}
    except Exception as e:
        logger.error(f"Get selections failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}/selections")
async def browser_clear_selections(session_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = await get_session_manager()
        success = await manager.clear_selections(session_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Clear selections failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/evaluate")
async def browser_evaluate(
    session_id: str, script: str = Form(...), tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = await get_session_manager()
        result = await manager.evaluate(session_id, script)
        return {"success": True, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Script evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
