import logging
from typing import Any

import pandas as pd
from openai import OpenAI
from rich import inspect
from rich.progress import track

from llm_lae.models import ExtractedData, Report

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

        extracted_data = self.extract_data_from_reports()
        self.export_extracted_data(extracted_data)

    def extract_data_from_reports(self) -> dict[str, ExtractedData]:
        extracted_data: dict[str, ExtractedData] = {}
        for report in track(
            self.reports,
            description="Extracting data from reports ...",
        ):
            logging.info(f"Extracting data of study ID: {report['study_id']}")

            data = self.extract_data_from_report(report["report_body"])
            extracted_data[report["study_id"]] = data

        return extracted_data

    def extract_data_from_report(self, report: str) -> ExtractedData:
        completion = self.client.beta.chat.completions.parse(
            model=self.open_ai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report},
            ],
            response_format=ExtractedData,
        )

        if self.debug:
            inspect(completion)

        usage = completion.usage
        assert usage
        logging.info(
            f"Total tokens: {usage.total_tokens}, "
            f"prompt tokens {usage.prompt_tokens}, "
            f"completion tokens {usage.completion_tokens}"
        )

        extracted_data = completion.choices[0].message.parsed
        assert extracted_data
        return extracted_data

    def export_extracted_data(self, extracted_data: dict[str, ExtractedData]) -> None:
        logging.info(f"Exporting extracted data to CSV file '{self.extracted_data_file}'.")

        items: list[dict[str, Any]] = []
        for study_id, data in extracted_data.items():
            item: dict[str, Any] = {"study_id": study_id}

            item = (
                item
                | data.clinical_information.model_dump()
                | data.indication.model_dump()
                | data.findings.model_dump()
            )

            item["keywords"] = ", ".join(data.clinical_information.keywords)
            item["clot_burden_score_calc"] = calc_cbs_score(data.findings)

            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.extracted_data_file, index=False)
