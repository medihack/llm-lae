import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from llm_lae.llm_extractor import LlmExtractor
from llm_lae.rules_extractor import RulesExtractor
from llm_lae.utils import sanitize_filename

from .generic_models import Report

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rules", action="store_true", help="Rule-based data extraction.")
    group.add_argument("--llm", help="LLM data extraction with the using the given model.")
    parser.add_argument("-l", "--limit", type=int, help="Limit the number of reports to process.")
    parser.add_argument(
        "-s",
        "--study-id",
        action="append",
        help="Only process report with the specified study ID.",
    )
    parser.add_argument(
        "-n",
        "--no-timestamp",
        action="store_true",
        help="Do not add timestamp prefix to output files.",
    )
    args = parser.parse_args()

    rules: bool = args.rules
    llm_model: str | None = args.llm

    limit_reports: int | None = None
    if args.limit is not None:
        limit_reports = args.limit

    study_ids: list[str] | None = None
    if args.study_id:
        study_ids = args.study_id

    reports_file = os.getenv("REPORTS_FILE")
    if not reports_file:
        raise ValueError("REPORTS_FILE environment variable is not set.")

    study_id_column = os.getenv("STUDY_ID_COLUMN")
    if not study_id_column:
        raise ValueError("STUDY_ID_COLUMN environment variable is not set.")

    report_column = os.getenv("REPORT_COLUMN")
    if not report_column:
        raise ValueError("REPORT_COLUMN environment variable is not set.")

    output_dir = os.getenv("OUTPUT_DIR")
    if not output_dir:
        raise ValueError("OUTPUT_DIR environment variable is not set.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.no_timestamp:
        timestamp = ""

    log_file = f"{output_dir}/{timestamp}_log.txt"
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

    if rules:
        input_values_file = f"{output_dir}/{timestamp}_input_values.csv"
        evaluated_values_file = f"{output_dir}/{timestamp}_evaluated_values.csv"

        RulesExtractor(
            reports=reports,
            input_values_file=input_values_file,
            evaluated_values_file=evaluated_values_file,
        ).extract()

    if llm_model:
        sanitized_filename = sanitize_filename(f"{timestamp}_extracted_{llm_model}.csv")
        extracted_data_file = f"{output_dir}/{sanitized_filename}"

        LlmExtractor(
            model=llm_model,
            reports=reports,
            extracted_data_file=extracted_data_file,
        ).extract()


if __name__ == "__main__":
    main()
