from openai import OpenAI
from config import CrashScenario
import os
import json
from dotenv import load_dotenv
load_dotenv()


class LLMParser:
    """Groq-based parser that extracts structured scene data from crash reports.

    Outputs:
    1. Basic crash fields (speed, weather, road, incident type)
    2. Scene objects with spatial positions (for depth map manipulation)
    3. Temporal description (for video generation)
    4. Rich image prompt (for diffusion model)
    """
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def parse(self, text: str) -> CrashScenario:
        """Extract structured scene data from a crash report.

        This is a CrashAgent-inspired approach: the LLM doesn't just extract
        flat fields — it reasons about the spatial layout of the scene,
        placing objects at estimated distances and positions.
        """
        system_prompt = """You are a crash scenario extraction system that produces structured scene representations.

Convert crash reports into JSON with these fields:

BASIC FIELDS:
- ego_speed_mps: speed in meters/second (convert mph * 0.44704)
- ego_action: "going_straight", "turning_left", "turning_right", "changing_lanes"
- weather: "clear", "rain", "snow", "fog", "night"
- lighting: "day", "dusk", "dawn", "night"
- road_type: "highway", "urban", "residential", "intersection"
- road_condition: "dry", "wet", "icy", "snow_covered"
- incident_type: "rear_end", "side_impact", "pedestrian", "cyclist", "hydroplane", "single_vehicle"
- severity: "minor", "moderate", "severe", "fatal"
- other_agent_type: null or "car", "truck", "pedestrian", "cyclist", "motorcycle"
- other_agent_action: null or "crossing", "stopped", "merging", "turning"

SCENE OBJECTS (NEW — spatial layout for depth map generation):
- scene_objects: array of objects in the scene. For each object:
  - type: "car", "truck", "pedestrian", "cyclist", "motorcycle", "guardrail", "debris"
  - distance_m: estimated distance from ego vehicle in meters (be realistic: pedestrian crossing = 10-30m, highway vehicle ahead = 30-100m)
  - lateral_position: -1.0 (far left) to 1.0 (far right), 0.0 = center of ego lane
  - width_fraction: object width as fraction of image (pedestrian ~0.03-0.05, car ~0.08-0.15, truck ~0.12-0.20)
  - height_fraction: object height as fraction of image (pedestrian ~0.15-0.25, car ~0.10-0.15, truck ~0.12-0.18)
  - action: what the object is doing ("crossing", "stopped", "merging", "turning", "approaching", "stationary")

Include ALL visible objects: the other vehicle/agent involved, parked cars, guardrails, road elements.

TEMPORAL DESCRIPTION:
- temporal_description: Describe how the scene evolves over 5 seconds leading to the incident. Example: "Vehicle approaches intersection at speed, cross-traffic enters from the right, collision occurs at center of intersection"

IMAGE PROMPT:
- image_prompt: A detailed Stable Diffusion prompt starting with "dashcam photo, point of view from inside a car, photorealistic, RAW photo, sharp focus" then describing the specific scene with weather, lighting, road, and object positions. Max 120 words.

Return ONLY valid JSON, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"crash report: {text}"}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")
            data = json.loads(content)
            data["description"] = text  # keep original for reference
            return CrashScenario(**data)  # Pydantic validates

        except Exception as e:
            raise RuntimeError(f"LLM parsing failed: {e}") from e


if __name__ == "__main__":
    parser = LLMParser()

    test_reports = [
        "Vehicle traveling 45mph on wet highway, hydroplaned into guardrail",
        "Pedestrian crossed outside crosswalk at night, struck by vehicle going 25mph",
        "Car rear-ended truck at stoplight in clear weather, going 30mph",
        "Side impact at intersection, vehicle ran red light at 35mph",
    ]

    for report in test_reports:
        print(f"\n--- Report: {report} ---")
        result = parser.parse(report)
        print(f"Objects in scene: {len(result.scene_objects)}")
        for obj in result.scene_objects:
            print(f"  {obj.type}: {obj.distance_m}m, lateral={obj.lateral_position}")
        print(f"Temporal: {result.temporal_description[:100]}...")
        print(f"Prompt: {result.image_prompt[:100]}...")
