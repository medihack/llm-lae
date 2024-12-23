import logging
from typing import Any

import pandas as pd
from openai import OpenAI
from rich import inspect
from rich.progress import track

from llm_lae.models import ExtractedData, LlmResult, Report

from .prompts import system_prompt
from .utils import calc_cbs_score


class LlmExtractor:
    def __init__(
        self,
        open_ai_base_url: str | None,
        open_ai_model: str,
        reports: list[Report],
        extracted_data_file: str,
        debug: bool,
    ) -> None:
        self.open_ai_base_url = open_ai_base_url
        self.open_ai_model = open_ai_model
        self.reports = reports
        self.extracted_data_file = extracted_data_file
        self.debug = debug

        self.client = OpenAI(base_url=self.open_ai_base_url)

    def extract(self) -> None:
        logging.info(f"Extracting data from reports with LLM model {self.open_ai_model}.")

        extracted_data = self.extract_from_reports()
        self.export_extracted_data(extracted_data)

    def extract_from_reports(self) -> list[LlmResult]:
        results: list[LlmResult] = []
        for report in track(
            self.reports,
            description="Extracting data from reports ...",
        ):
            logging.info(f"Extracting data of study ID: {report['study_id']}")

            result = self.extract_from_report(report)
            results.append(result)

        return results

    def extract_from_report(self, report: Report) -> LlmResult:
        completion = self.client.beta.chat.completions.parse(
            model=self.open_ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report["report_body"]},
            ],
            response_format=ExtractedData,
            temperature=0.0,
        )

        if self.debug:
            inspect(completion)

        extracted_data = completion.choices[0].message.parsed
        assert extracted_data

        usage = completion.usage
        assert usage

        logging.info(
            f"Total tokens: {usage.total_tokens}, "
            f"prompt tokens {usage.prompt_tokens}, "
            f"completion tokens {usage.completion_tokens}"
        )

        result = LlmResult(
            extracted_data=extracted_data,
            study_id=report["study_id"],
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

        return result

    def export_extracted_data(self, results: list[LlmResult]) -> None:
        logging.info(f"Exporting extracted data to CSV file '{self.extracted_data_file}'.")

        items: list[dict[str, Any]] = []
        for result in results:
            item: dict[str, Any] = {"study_id": result["study_id"]}

            item = (
                item
                | result["extracted_data"].clinical_information.model_dump()
                | result["extracted_data"].indication.model_dump()
                | result["extracted_data"].findings.model_dump()
            )

            item["keywords"] = ", ".join(result["extracted_data"].clinical_information.keywords)
            item["clot_burden_score_calc"] = calc_cbs_score(result["extracted_data"].findings)

            item["total_tokens"] = result["total_tokens"]
            item["prompt_tokens"] = result["prompt_tokens"]
            item["completion_tokens"] = result["completion_tokens"]

            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.extracted_data_file, index=False)
