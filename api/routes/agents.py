"""
Agent (async job) routes — create, manage, schedule scraping agents.
"""
import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel, Field

from api.middleware.auth import get_tenant_id
from universal_scraper.core.agent_manager import AgentManager, AgentType, AgentStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


class CreateAgentRequest(BaseModel):
    type: str = Field(..., description="Agent type: 'web_scraping', 'document_processing', 'batch_scraping'")
    config: Dict[str, Any] = Field(..., description="Agent configuration")
    queue_immediately: bool = Field(default=True, description="Queue for execution immediately")


class ScheduleAgentRequest(BaseModel):
    schedule: str = Field(..., description="Cron expression (e.g., '0 */6 * * *' for every 6 hours)")
    timezone: str = Field(default="UTC", description="Timezone for schedule")


class CreateAgentFromCacheRequest(BaseModel):
    domain: str = Field(..., description="Domain from cache")
    fields: List[str] = Field(..., description="Fields to extract")
    url: str = Field(..., description="URL to scrape")
    visibility: str = Field(default="private", description="Cache visibility")
    schedule: Optional[str] = Field(default=None, description="Optional cron schedule")


def _get_agent_manager():
    from api.main import get_agent_manager
    return get_agent_manager()


def _execute_agent_task(agent_id, api_key):
    from api.main import execute_agent_task
    return execute_agent_task(agent_id, api_key)


@router.post("")
async def create_agent(
    request: CreateAgentRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    try:
        manager = _get_agent_manager()
        agent_type = AgentType(request.type)
        agent = await manager.create_agent(
            tenant_id=tenant_id, agent_type=agent_type,
            config=request.config, queue_immediately=request.queue_immediately,
        )
        if request.queue_immediately and not manager.cloud_tasks_enabled:
            background_tasks.add_task(_execute_agent_task, agent.id, x_api_key)
        return {"success": True, "agent": agent.to_dict()}
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_agents(
    status: Optional[str] = None, type: Optional[str] = None,
    limit: int = 50, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = _get_agent_manager()
        status_filter = AgentStatus(status) if status else None
        type_filter = AgentType(type) if type else None
        agents = await manager.list_agents(tenant_id, status_filter, type_filter, limit)
        return {"success": True, "agents": [a.to_dict() for a in agents], "total": len(agents)}
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_agent_stats(tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = _get_agent_manager()
        stats = await manager.get_stats(tenant_id)
        return {"success": True, **stats}
    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}")
async def get_agent(agent_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = _get_agent_manager()
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"success": True, "agent": agent.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/cancel")
async def cancel_agent(agent_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = _get_agent_manager()
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        success = await manager.cancel_agent(agent_id)
        return {"success": success, "message": "Agent cancelled" if success else "Cannot cancel agent in current state"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/execute")
async def execute_agent_endpoint(
    agent_id: str, x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    try:
        await _execute_agent_task(agent_id, x_api_key)
        return {"success": True, "message": "Agent execution started"}
    except Exception as e:
        logger.error(f"Failed to execute agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/schedule")
async def schedule_agent(
    agent_id: str, request: ScheduleAgentRequest, tenant_id: str = Depends(get_tenant_id),
):
    try:
        manager = _get_agent_manager()
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        success = await manager.schedule_agent(agent_id, request.schedule, request.timezone)
        return {"success": success, "message": f"Agent scheduled: {request.schedule}" if success else "Failed to schedule agent"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}/schedule")
async def unschedule_agent(agent_id: str, tenant_id: str = Depends(get_tenant_id)):
    try:
        manager = _get_agent_manager()
        agent = await manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        success = await manager.unschedule_agent(agent_id)
        return {"success": success, "message": "Schedule removed" if success else "Agent was not scheduled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unschedule agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-cache")
async def create_agent_from_cache(
    request: CreateAgentFromCacheRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    try:
        manager = _get_agent_manager()
        agent = await manager.create_from_cache(
            tenant_id=tenant_id, domain=request.domain, fields=request.fields,
            url=request.url, visibility=request.visibility, schedule=request.schedule,
        )
        if not request.schedule and not manager.cloud_tasks_enabled:
            background_tasks.add_task(_execute_agent_task, agent.id, x_api_key)
        return {
            "success": True, "agent": agent.to_dict(),
            "message": f"Agent created from cache ({request.domain})" + (f" with schedule: {request.schedule}" if request.schedule else ""),
        }
    except Exception as e:
        logger.error(f"Failed to create agent from cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
