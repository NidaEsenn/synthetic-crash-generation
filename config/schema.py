from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    NIGHT = "night"
    # ... add the rest

class IncidentType(str, Enum):
    REAR_END = "rear_end"
    SIDE_IMPACT = "side_impact"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    HYDROPLANE = "hydroplane"
    SINGLE_VEHICLE = "single_vehicle"
    # ... add the rest

class CrashScenario(BaseModel):
    """Structured crash scenario for generation control"""
    # Ego vehicle (the car we're simulating)
    ego_speed_mps: float = Field(..., description="Speed in meters/second")
    ego_action: str = Field(default="going_straight")
    
    # Environment
    weather: Weather
    lighting: str = Field(default="day")
    road_type: str = Field(default="highway")
    road_condition: str = Field(default="dry")
    
    #Incident details
    incident_type: IncidentType
    severity: str = Field(default="moderate")
    # ... add remaining fields from the implementation.md
    # Other agents
    other_agent_type: Optional[str] = None
    other_agent_action: Optional[str] = None
    
    # Metadata
    description: str = Field(default="", description="Original text description")
    image_prompt: str = Field(default="", description="Rich Stable Diffusion prompt for image generation")

    class Config:
        use_enum_values = True  # stores "rain" not Weather.RAIN
