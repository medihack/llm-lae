import os
from timeit import default_timer as timer

from ollama import Client
from openai import OpenAI

from .conf import OPEN_AI_MODELS
from .generic_models import Report
from .llm_models import ExtractedData, LlmResult
from .llm_prompts import SYSTEM_PROMPT


class LlmClient:
    def __init__(self, model: str) -> None:
        self.model = model
        if model in OPEN_AI_MODELS:
            self.client = OpenAI()
        else:
            self.client = Client(host=os.getenv("OLLAMA_HOST"))

    def extract(self, report: Report) -> LlmResult:
        if isinstance(self.client, OpenAI):
            return self.extract_with_openai(self.client, report)
        else:
            return self.extract_with_ollama(self.client, report)

    def extract_with_openai(self, client: OpenAI, report: Report) -> LlmResult:
        start = timer()
        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": report["report_body"]},
            ],
            temperature=0.0,
            response_format=ExtractedData,
        )
        end = timer()

        extracted_data = completion.choices[0].message.parsed
        assert extracted_data

        usage = completion.usage
        assert usage

        return LlmResult(
            extracted_data=extracted_data,
            study_id=report["study_id"],
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            duration=end - start,
        )

    def extract_with_ollama(self, client: Client, report: Report) -> LlmResult:
        start = timer()
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": report["report_body"]},
            ],
            options={"temperature": 0.0, "num_ctx": 8192},
            format=ExtractedData.model_json_schema(),
        )
        end = timer()

        assert response.message.content is not None
        extracted_data = ExtractedData.model_validate_json(response.message.content)

        assert response.prompt_eval_count is not None
        assert response.eval_count is not None
        return LlmResult(
            extracted_data=extracted_data,
            study_id=report["study_id"],
            total_tokens=response.prompt_eval_count + response.eval_count,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            duration=end - start,
        )
