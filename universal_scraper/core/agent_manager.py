"""
Agent Manager - Handles async job execution for scraping and document processing
Uses Cloud Tasks for reliable, scalable job queue processing
"""
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    WEB_SCRAPING = "web_scraping"
    DOCUMENT_PROCESSING = "document_processing"
    BATCH_SCRAPING = "batch_scraping"


@dataclass
class Agent:
    """Represents a scraping/processing job"""
    id: str
    tenant_id: str
    type: AgentType
    status: AgentStatus
    config: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: int = 0
    progress_message: str = ""
    # Scheduling fields
    schedule: Optional[str] = None  # Cron expression (e.g., "0 */6 * * *" for every 6 hours)
    schedule_id: Optional[str] = None  # Cloud Scheduler job ID
    next_run: Optional[float] = None  # Next scheduled run timestamp
    last_run: Optional[float] = None  # Last run timestamp
    run_count: int = 0  # Number of times this agent has run
    # Cache reference (if created from cache)
    from_cache: bool = False
    cache_domain: Optional[str] = None
    cache_visibility: Optional[str] = None  # 'public' or 'private'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "type": self.type.value,
            "status": self.status.value,
            "config": self.config,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "schedule": self.schedule,
            "schedule_id": self.schedule_id,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "from_cache": self.from_cache,
            "cache_domain": self.cache_domain,
            "cache_visibility": self.cache_visibility,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        return cls(
            id=data["id"],
            tenant_id=data["tenant_id"],
            type=AgentType(data["type"]),
            status=AgentStatus(data["status"]),
            config=data["config"],
            result=data.get("result"),
            error=data.get("error"),
            created_at=data.get("created_at", 0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            progress=data.get("progress", 0),
            progress_message=data.get("progress_message", ""),
            schedule=data.get("schedule"),
            schedule_id=data.get("schedule_id"),
            next_run=data.get("next_run"),
            last_run=data.get("last_run"),
            run_count=data.get("run_count", 0),
            from_cache=data.get("from_cache", False),
            cache_domain=data.get("cache_domain"),
            cache_visibility=data.get("cache_visibility"),
        )


class AgentManager:
    """
    Manages agent (job) lifecycle

    Features:
    - Create and queue agents
    - Track agent status and progress
    - Store results
    - Support for Cloud Tasks integration
    """

    def __init__(self, redis_cache: Optional[Any] = None, cloud_tasks_enabled: bool = False):
        """
        Initialize agent manager

        Args:
            redis_cache: RedisCache instance for job storage
            cloud_tasks_enabled: Whether to use Cloud Tasks for queuing
        """
        self.redis_cache = redis_cache
        self.cloud_tasks_enabled = cloud_tasks_enabled
        self.prefix = "agent:"
        self.list_prefix = "agent_list:"

        if redis_cache and redis_cache.redis_client:
            logger.info("AgentManager initialized with Redis backend")
        else:
            logger.warning("AgentManager initialized without Redis - agents won't persist")

    def _make_key(self, agent_id: str) -> str:
        """Generate cache key for agent"""
        return f"{self.prefix}{agent_id}"

    def _make_list_key(self, tenant_id: str) -> str:
        """Generate list key for tenant's agents"""
        return f"{self.list_prefix}{tenant_id}"

    async def create_agent(
        self,
        tenant_id: str,
        agent_type: AgentType,
        config: Dict[str, Any],
        queue_immediately: bool = True
    ) -> Agent:
        """
        Create a new agent (job)

        Args:
            tenant_id: Tenant identifier
            agent_type: Type of agent
            config: Agent configuration (URL, fields, etc.)
            queue_immediately: Whether to queue for execution immediately

        Returns:
            Created Agent
        """
        agent_id = str(uuid.uuid4())

        agent = Agent(
            id=agent_id,
            tenant_id=tenant_id,
            type=agent_type,
            status=AgentStatus.PENDING,
            config=config,
            created_at=time.time(),
        )

        # Store agent
        await self._store_agent(agent)

        # Add to tenant's agent list
        await self._add_to_list(tenant_id, agent_id)

        if queue_immediately:
            await self.queue_agent(agent_id)

        logger.info(f"Created agent {agent_id} for tenant {tenant_id}")
        return agent

    async def _store_agent(self, agent: Agent) -> bool:
        """Store agent in Redis"""
        if not self.redis_cache:
            return False

        try:
            key = self._make_key(agent.id)
            await self.redis_cache.set(key, agent.to_dict(), ttl=86400 * 7)  # 7 days
            return True
        except Exception as e:
            logger.error(f"Failed to store agent: {e}")
            return False

    async def _add_to_list(self, tenant_id: str, agent_id: str) -> bool:
        """Add agent to tenant's list"""
        if not self.redis_cache or not self.redis_cache.redis_client:
            return False

        try:
            list_key = self._make_list_key(tenant_id)
            await self.redis_cache.redis_client.lpush(list_key, agent_id)
            # Keep only last 100 agents per tenant
            await self.redis_cache.redis_client.ltrim(list_key, 0, 99)
            return True
        except Exception as e:
            logger.error(f"Failed to add agent to list: {e}")
            return False

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        if not self.redis_cache:
            return None

        try:
            key = self._make_key(agent_id)
            data = await self.redis_cache.get(key)
            if data:
                return Agent.from_dict(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get agent: {e}")
            return None

    async def list_agents(
        self,
        tenant_id: str,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        limit: int = 50
    ) -> List[Agent]:
        """
        List agents for tenant

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            agent_type: Optional type filter
            limit: Maximum number of agents to return

        Returns:
            List of agents
        """
        if not self.redis_cache or not self.redis_cache.redis_client:
            return []

        try:
            list_key = self._make_list_key(tenant_id)
            agent_ids = await self.redis_cache.redis_client.lrange(list_key, 0, limit - 1)

            agents = []
            for agent_id in agent_ids:
                agent = await self.get_agent(agent_id)
                if agent:
                    # Apply filters
                    if status and agent.status != status:
                        continue
                    if agent_type and agent.type != agent_type:
                        continue
                    agents.append(agent)

            return agents

        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []

    async def queue_agent(self, agent_id: str) -> bool:
        """
        Queue agent for execution

        Args:
            agent_id: Agent ID to queue

        Returns:
            True if queued successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        if self.cloud_tasks_enabled:
            # Queue via Cloud Tasks
            success = await self._queue_cloud_task(agent)
            if success:
                agent.status = AgentStatus.QUEUED
                await self._store_agent(agent)
                return True
        else:
            # Direct execution (for development/testing)
            agent.status = AgentStatus.QUEUED
            await self._store_agent(agent)
            return True

        return False

    async def _queue_cloud_task(self, agent: Agent) -> bool:
        """Queue agent via Cloud Tasks"""
        try:
            from google.cloud import tasks_v2
            import os

            project = os.getenv("GOOGLE_CLOUD_PROJECT", "soma-data-467016")
            location = os.getenv("CLOUD_TASKS_LOCATION", "us-central1")
            queue = os.getenv("CLOUD_TASKS_QUEUE", "paradocs-agents")
            service_url = os.getenv("SERVICE_URL", "https://universal-scraper-api-r3crozpq7q-uc.a.run.app")

            client = tasks_v2.CloudTasksClient()
            parent = client.queue_path(project, location, queue)

            # Create task
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": f"{service_url}/api/v1/agents/{agent.id}/execute",
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"agent_id": agent.id}).encode(),
                }
            }

            response = client.create_task(request={"parent": parent, "task": task})
            logger.info(f"Created Cloud Task for agent {agent.id}: {response.name}")
            return True

        except ImportError:
            logger.warning("Cloud Tasks library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to queue Cloud Task: {e}")
            return False

    async def update_progress(
        self,
        agent_id: str,
        progress: int,
        message: str = ""
    ) -> bool:
        """
        Update agent progress

        Args:
            agent_id: Agent ID
            progress: Progress percentage (0-100)
            message: Progress message

        Returns:
            True if updated successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        agent.progress = progress
        agent.progress_message = message

        if progress > 0 and agent.status == AgentStatus.QUEUED:
            agent.status = AgentStatus.RUNNING
            agent.started_at = time.time()

        return await self._store_agent(agent)

    async def complete_agent(
        self,
        agent_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Mark agent as completed with result

        Args:
            agent_id: Agent ID
            result: Execution result

        Returns:
            True if updated successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.COMPLETED
        agent.result = result
        agent.completed_at = time.time()
        agent.progress = 100
        agent.progress_message = "Completed"

        return await self._store_agent(agent)

    async def fail_agent(
        self,
        agent_id: str,
        error: str
    ) -> bool:
        """
        Mark agent as failed with error

        Args:
            agent_id: Agent ID
            error: Error message

        Returns:
            True if updated successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.FAILED
        agent.error = error
        agent.completed_at = time.time()

        return await self._store_agent(agent)

    async def cancel_agent(self, agent_id: str) -> bool:
        """
        Cancel a pending/queued agent

        Args:
            agent_id: Agent ID

        Returns:
            True if cancelled successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        if agent.status in [AgentStatus.PENDING, AgentStatus.QUEUED]:
            agent.status = AgentStatus.CANCELLED
            agent.completed_at = time.time()
            return await self._store_agent(agent)

        return False

    async def get_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get agent statistics for tenant

        Args:
            tenant_id: Tenant identifier

        Returns:
            Statistics dict
        """
        agents = await self.list_agents(tenant_id, limit=100)

        status_counts = {}
        type_counts = {}
        scheduled_count = 0

        for agent in agents:
            status_counts[agent.status.value] = status_counts.get(agent.status.value, 0) + 1
            type_counts[agent.type.value] = type_counts.get(agent.type.value, 0) + 1
            if agent.schedule:
                scheduled_count += 1

        return {
            "total_agents": len(agents),
            "by_status": status_counts,
            "by_type": type_counts,
            "scheduled_count": scheduled_count,
        }

    async def schedule_agent(
        self,
        agent_id: str,
        schedule: str,  # Cron expression
        timezone: str = "UTC"
    ) -> bool:
        """
        Schedule an agent to run periodically using Cloud Scheduler

        Args:
            agent_id: Agent ID
            schedule: Cron expression (e.g., "0 */6 * * *" for every 6 hours)
            timezone: Timezone for schedule (default: UTC)

        Returns:
            True if scheduled successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return False

        try:
            from google.cloud import scheduler_v1
            import os

            project = os.getenv("GOOGLE_CLOUD_PROJECT", "soma-data-467016")
            location = os.getenv("CLOUD_SCHEDULER_LOCATION", "us-central1")
            service_url = os.getenv("SERVICE_URL", "https://universal-scraper-api-r3crozpq7q-uc.a.run.app")

            client = scheduler_v1.CloudSchedulerClient()
            parent = f"projects/{project}/locations/{location}"

            job_id = f"agent-{agent_id[:8]}"
            job_name = f"{parent}/jobs/{job_id}"

            # Create scheduler job
            job = {
                "name": job_name,
                "description": f"Scheduled agent: {agent.config.get('url', 'Unknown')}",
                "schedule": schedule,
                "time_zone": timezone,
                "http_target": {
                    "uri": f"{service_url}/api/v1/agents/{agent_id}/execute",
                    "http_method": scheduler_v1.HttpMethod.POST,
                    "headers": {"Content-Type": "application/json"},
                },
            }

            # Try to create or update
            try:
                response = client.create_job(request={"parent": parent, "job": job})
                logger.info(f"Created Cloud Scheduler job: {response.name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    response = client.update_job(request={"job": job})
                    logger.info(f"Updated Cloud Scheduler job: {response.name}")
                else:
                    raise

            # Update agent with schedule info
            agent.schedule = schedule
            agent.schedule_id = job_id
            await self._store_agent(agent)

            return True

        except ImportError:
            logger.warning("Cloud Scheduler library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to schedule agent: {e}")
            return False

    async def unschedule_agent(self, agent_id: str) -> bool:
        """
        Remove schedule from an agent

        Args:
            agent_id: Agent ID

        Returns:
            True if unscheduled successfully
        """
        agent = await self.get_agent(agent_id)
        if not agent or not agent.schedule_id:
            return False

        try:
            from google.cloud import scheduler_v1
            import os

            project = os.getenv("GOOGLE_CLOUD_PROJECT", "soma-data-467016")
            location = os.getenv("CLOUD_SCHEDULER_LOCATION", "us-central1")

            client = scheduler_v1.CloudSchedulerClient()
            job_name = f"projects/{project}/locations/{location}/jobs/{agent.schedule_id}"

            client.delete_job(request={"name": job_name})
            logger.info(f"Deleted Cloud Scheduler job: {job_name}")

            # Update agent
            agent.schedule = None
            agent.schedule_id = None
            agent.next_run = None
            await self._store_agent(agent)

            return True

        except ImportError:
            logger.warning("Cloud Scheduler library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to unschedule agent: {e}")
            return False

    async def create_from_cache(
        self,
        tenant_id: str,
        domain: str,
        fields: List[str],
        url: str,
        visibility: str = "private",
        schedule: Optional[str] = None
    ) -> Agent:
        """
        Create an agent from a cached pattern

        Args:
            tenant_id: Tenant identifier
            domain: Domain from cache
            fields: Fields to extract
            url: URL to scrape
            visibility: Cache visibility ('public' or 'private')
            schedule: Optional cron schedule

        Returns:
            Created Agent
        """
        config = {
            "url": url,
            "fields": fields,
            "mode": "hybrid",
            "from_cache_domain": domain,
        }

        agent = await self.create_agent(
            tenant_id=tenant_id,
            agent_type=AgentType.WEB_SCRAPING,
            config=config,
            queue_immediately=schedule is None  # Don't queue if scheduled
        )

        # Set cache reference
        agent.from_cache = True
        agent.cache_domain = domain
        agent.cache_visibility = visibility

        # Set schedule if provided
        if schedule:
            agent.schedule = schedule
            await self._store_agent(agent)
            await self.schedule_agent(agent.id, schedule)
        else:
            await self._store_agent(agent)

        return agent

