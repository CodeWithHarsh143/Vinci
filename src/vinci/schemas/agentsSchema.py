from pydantic import BaseModel


class PlanModel(BaseModel):
    plan: list[str]
    dependencies: list[list[str | None]]
    estimated_time_minutes: int
