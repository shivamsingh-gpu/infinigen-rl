"""Restricted action space for the beginner loop.

Keep this tiny on purpose. The point is to make the LLM's job well-defined and
the reward signal legible. Widen only after the loop runs end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import json
import random

Biome = Literal["forest", "desert", "mountain", "coast", "arctic"]
TimeOfDay = Literal["dawn", "noon", "golden_hour", "dusk", "night"]
Weather = Literal["clear", "overcast", "foggy", "rain"]

BIOMES: tuple[Biome, ...] = ("forest", "desert", "mountain", "coast", "arctic")
TIMES: tuple[TimeOfDay, ...] = ("dawn", "noon", "golden_hour", "dusk", "night")
WEATHERS: tuple[Weather, ...] = ("clear", "overcast", "foggy", "rain")


@dataclass
class SceneConfig:
    biome: Biome
    time_of_day: TimeOfDay
    weather: Weather
    camera_height_m: float
    camera_pitch_deg: float
    vegetation_density: float
    seed: int

    def __post_init__(self):
        # The policy emits arbitrary integers here; Infinigen requires a uint32
        # (0 <= seed <= 2**32 - 1) and hard-crashes otherwise. Fold any int into
        # range so a valid rollout is never wasted on an out-of-range seed.
        self.seed = int(self.seed) % (2 ** 32)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "SceneConfig":
        d = json.loads(s)
        return cls(**d)

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "SceneConfig":
        r = rng or random
        return cls(
            biome=r.choice(BIOMES),
            time_of_day=r.choice(TIMES),
            weather=r.choice(WEATHERS),
            camera_height_m=round(r.uniform(1.2, 8.0), 2),
            camera_pitch_deg=round(r.uniform(-20.0, 10.0), 1),
            vegetation_density=round(r.uniform(0.0, 1.0), 2),
            seed=r.randint(0, 2**31 - 1),
        )


SCHEMA_FOR_LLM = """\
Return ONLY a JSON object with these keys and constraints:
- biome: one of ["forest", "desert", "mountain", "coast", "arctic"]
- time_of_day: one of ["dawn", "noon", "golden_hour", "dusk", "night"]
- weather: one of ["clear", "overcast", "foggy", "rain"]
- camera_height_m: float in [1.2, 8.0]
- camera_pitch_deg: float in [-20.0, 10.0]  (negative = looking down)
- vegetation_density: float in [0.0, 1.0]
- seed: integer in [0, 2147483647]
"""
