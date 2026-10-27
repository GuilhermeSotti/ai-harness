import asyncio
import json

import pytest

from ai_harness.planning import create_dynamic_plan
from ai_harness.providers import GenerationResult


class PlanProvider:
    provider_id = "test"
    model = "plan-test"

    def __init__(self, response):
        self.response = response
        self.prompt = None

    async def generate_json(self, *, system, prompt, schema):
        self.prompt = prompt
        return GenerationResult(
            text=json.dumps(self.response),
            duration_ms=2,
            input_tokens=10,
            output_tokens=20,
        )

    async def health(self):
        raise AssertionError


def test_dynamic_plan_validates_dependencies_and_converts_indexes():
    provider = PlanProvider({
        "tasks": [
            {
                "description": "Analyze implementation evidence.",
                "required_capability": "analysis",
                "depends_on": [],
            },
            {
                "description": "Independently review findings.",
                "required_capability": "review",
                "depends_on": [0],
            },
        ]
    })

    plan = asyncio.run(create_dynamic_plan(provider, "Review this project."))

    assert len(plan.tasks) == 2
    assert plan.tasks[1].dependencies == (plan.tasks[0].task_id,)
    assert provider.prompt == "Review this project."


@pytest.mark.parametrize(
    "response",
    [
        {"tasks": []},
        {"tasks": [{
            "description": "Invalid dependency",
            "required_capability": "analysis",
            "depends_on": [5],
        }]},
        {"tasks": [
            {"description": "A", "required_capability": "analysis", "depends_on": [1]},
            {"description": "B", "required_capability": "review", "depends_on": [0]},
        ]},
    ],
)
def test_invalid_provider_plans_are_rejected(response):
    with pytest.raises(ValueError, match="invalid task plan|cyclic plan"):
        asyncio.run(create_dynamic_plan(PlanProvider(response), "Do work."))


def test_blank_objective_is_rejected_before_provider_call():
    provider = PlanProvider({"tasks": []})

    with pytest.raises(ValueError, match="blank"):
        asyncio.run(create_dynamic_plan(provider, "  "))

    assert provider.prompt is None
