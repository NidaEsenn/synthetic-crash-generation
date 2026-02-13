from openai import OpenAI
from config import CrashScenario
import os
import json
from dotenv import load_dotenv
load_dotenv()

class LLMParser:
    """Groq-based parser for complex reports"""
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def parse(self, text: str) -> CrashScenario:
        """Extract structured data AND generate a rich image prompt from crash report"""
        system_prompt = """You are a crash scenario extraction system.
        Convert crash reports into structured JSON with these exact fields:
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
        - image_prompt: A detailed Stable Diffusion prompt for generating a photorealistic dashcam image of this exact scenario. Must start with "dashcam photo, point of view from inside a car, photorealistic, RAW photo, sharp focus" then describe the SPECIFIC scene: road environment, weather/lighting atmosphere, what is visible ahead, specific positions of vehicles/pedestrians, time of day details. Be very specific to the crash report — do NOT use generic descriptions. Max 120 words.

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
            content=response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")
            data = json.loads(content)
            data["description"] =text # keep original for reference
            return CrashScenario(**data)  # Pydantic validates it

        except Exception as e:
            raise RuntimeError(f"LLM parsing failed: {e}") from e


# Test
if __name__ == "__main__":
    parser = LLMParser()
    test = "Vehicle traveling 30mph in residential area, child ran into street from behind parked car"
    result = parser.parse(test)
    print(result.model_dump_json(indent=2))