import logging
from typing import Any

import pandas as pd
from rich.progress import track

from llm_lae.llm_client import LlmClient

from .generic_models import Report
from .llm_models import LlmResult
from .utils import calc_cbs_score


class LlmExtractor:
    def __init__(
        self,
        model: str,
        reports: list[Report],
        extracted_data_file: str,
    ) -> None:
        self.model = model
        self.reports = reports
        self.extracted_data_file = extracted_data_file

        self.client = LlmClient(model)

    def extract(self) -> None:
        logging.info(f"Extracting data from reports with LLM model '{self.model}'.")

        extracted_data = self.extract_from_reports()
        self.export_extracted_data(extracted_data)

    def extract_from_reports(self) -> list[LlmResult]:
        results: list[LlmResult] = []
        for report in track(
            self.reports,
            description="Extracting data from reports ...",
        ):
            logging.info(f"Extracting data of study ID: {report['study_id']}")

            try:
                results.append(self.client.extract(report))
            except Exception:
                logging.exception(f"Failed to extract data of study {report['study_id']}.")

        return results

    def export_extracted_data(self, results: list[LlmResult]) -> None:
        logging.info(f"Exporting LLM extracted data to CSV file '{self.extracted_data_file}'.")

        items: list[dict[str, Any]] = []
        for result in results:
            item: dict[str, Any] = {"study_id": result.study_id}

            item = (
                item
                | result.extracted_data.clinical_information.model_dump()
                | result.extracted_data.indication.model_dump()
                | result.extracted_data.findings.model_dump()
            )

            item["keywords"] = ", ".join(result.extracted_data.clinical_information.keywords)
            item["clot_burden_score_calc"] = calc_cbs_score(result.extracted_data.findings)

            item["total_tokens"] = result.total_tokens
            item["prompt_tokens"] = result.prompt_tokens
            item["completion_tokens"] = result.completion_tokens
            item["duration"] = result.duration

            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.extracted_data_file, index=False)
