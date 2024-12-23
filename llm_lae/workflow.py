import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv

from llm_lae.input_validator import InputValidator
from llm_lae.llm_extractor import LlmExtractor

from .models import Report

load_dotenv()


def setup_logging(log_file: str):
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int)
    parser.add_argument("-s", "--study-id", action="append")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    limit_reports: int | None = None
    if args.limit is not None:
        limit_reports = args.limit

    study_ids: list[str] | None = None
    if args.study_id:
        study_ids = args.study_id

    debug = False
    if args.debug:
        debug = True

    open_ai_base_url = os.getenv("OPENAI_BASE_URL")

    open_ai_model = os.getenv("OPENAI_MODEL")
    if not open_ai_model:
        raise ValueError("OPENAI_MODEL environment variable is not set.")

    log_file = os.getenv("LOG_FILE")
    if not log_file:
        raise ValueError("LOG_FILE environment variable is not set.")

    reports_file = os.getenv("REPORTS_FILE")
    if not reports_file:
        raise ValueError("REPORTS_FILE environment variable is not set.")

    validations_file = os.getenv("VALIDATIONS_FILE")
    if not validations_file:
        raise ValueError("VALIDATIONS_FILE environment variable is not set.")

    extracted_data_file = os.getenv("EXTRACTED_DATA_FILE")
    if not extracted_data_file:
        raise ValueError("EXTRACTED_DATA_FILE environment variable is not set.")

    study_id_column = os.getenv("STUDY_ID_COLUMN")
    if not study_id_column:
        raise ValueError("STUDY_ID_COLUMN environment variable is not set.")

    report_column = os.getenv("REPORT_COLUMN")
    if not report_column:
        raise ValueError("REPORT_COLUMN environment variable is not set.")

    setup_logging(log_file)

    logging.info("Loading data from CSV file.")
    df = pd.read_csv(reports_file)
    if study_ids is not None:
        df = pd.DataFrame(df[df[study_id_column].isin(study_ids)])
    elif isinstance(limit_reports, int):
        df = df.head(limit_reports)

    reports: list[Report] = []
    for _, row in df.iterrows():
        study_id = str(row[study_id_column])
        report = str(row[report_column])
        reports.append(Report(study_id=study_id, report_body=report))

    InputValidator(reports=reports, validations_file=validations_file).validate()
    LlmExtractor(
        open_ai_base_url=open_ai_base_url,
        open_ai_model=open_ai_model,
        reports=reports,
        extracted_data_file=extracted_data_file,
        debug=debug,
    ).extract()


if __name__ == "__main__":
    main()
