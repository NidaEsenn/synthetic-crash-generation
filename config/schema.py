from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    NIGHT = "night"


class IncidentType(str, Enum):
    REAR_END = "rear_end"
    SIDE_IMPACT = "side_impact"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    HYDROPLANE = "hydroplane"
    SINGLE_VEHICLE = "single_vehicle"


class SceneObject(BaseModel):
    """A single object in the crash scene with spatial information.

    Used to drive depth map manipulation — each object's position
    controls where ControlNet places it in the generated image.
    """
    type: str = Field(..., description="Object type: car, truck, pedestrian, cyclist, motorcycle, guardrail, debris")
    distance_m: float = Field(..., description="Distance from ego vehicle in meters")
    lateral_position: float = Field(
        default=0.0,
        description="Horizontal position: -1.0 (far left) to 1.0 (far right), 0.0 = center"
    )
    width_fraction: float = Field(
        default=0.05,
        description="Approximate width as fraction of image (0.0-1.0)"
    )
    height_fraction: float = Field(
        default=0.15,
        description="Approximate height as fraction of image (0.0-1.0)"
    )
    action: Optional[str] = Field(
        default=None,
        description="What the object is doing: crossing, stopped, merging, turning, approaching"
    )


class CrashScenario(BaseModel):
    """Structured crash scenario for generation control.

    This is the single source of truth for a crash scenario. It contains:
    1. Basic fields (ego vehicle, environment, incident) — original schema
    2. Scene layout (object positions) — for depth map manipulation
    3. Image prompt — for diffusion model text conditioning
    """
    # Ego vehicle (the car we're simulating)
    ego_speed_mps: float = Field(..., description="Speed in meters/second")
    ego_action: str = Field(default="going_straight")

    # Environment
    weather: Weather
    lighting: str = Field(default="day")
    road_type: str = Field(default="highway")
    road_condition: str = Field(default="dry")

    # Incident details
    incident_type: IncidentType
    severity: str = Field(default="moderate")

    # Other agents (legacy fields — kept for backward compatibility)
    other_agent_type: Optional[str] = None
    other_agent_action: Optional[str] = None

    # Scene layout — list of objects with spatial positions
    scene_objects: list[SceneObject] = Field(
        default_factory=list,
        description="Objects in the scene with positions for depth map manipulation"
    )

    # Temporal description for video generation
    temporal_description: str = Field(
        default="",
        description="How the scene evolves over 5 seconds: what approaches, what changes"
    )

    # Metadata
    description: str = Field(default="", description="Original text description")
    image_prompt: str = Field(default="", description="Rich Stable Diffusion prompt for image generation")

    class Config:
        use_enum_values = True  # stores "rain" not Weather.RAIN
