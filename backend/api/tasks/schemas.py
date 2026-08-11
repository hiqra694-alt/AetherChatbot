from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"


class Task(BaseModel):
    """
    A single user-owned task/reminder row in `tasks`. When `due_at` is set,
    the background scheduler (core/scheduler.py) flags this task (via
    `notified_at`) once due_at has passed while it's still pending.
    """
    id: str = Field(..., description="Task row id")
    user_id: str = Field(..., description="Id of the user who owns this task")
    title: str = Field(..., description="Short title describing the task")
    description: Optional[str] = Field(None, description="Optional longer description of the task")
    due_at: Optional[datetime] = Field(None, description="When the task is due, if any")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current task status")
    created_at: datetime = Field(..., description="When the task was created")
    notified_at: Optional[datetime] = Field(
        None, description="When the scheduler last flagged this task as due, if ever"
    )

    @field_validator("due_at", "notified_at", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value


class TaskListResponse(BaseModel):
    """Payload for GET /api/tasks."""
    tasks: List[Task] = Field(default_factory=list)


class TaskCompleteResponse(BaseModel):
    """Payload for PATCH /api/tasks/{task_id}/complete."""
    completed: bool = Field(..., description="Whether a matching task existed and was marked completed")


class TaskDeleteResponse(BaseModel):
    """Payload for DELETE /api/tasks/{task_id}."""
    deleted: bool = Field(..., description="Whether a matching task existed and was removed")
