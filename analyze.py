import argparse
import logging
import os
import re
from enum import Enum
from typing import Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.progress import track

load_dotenv()

SYSTEM_PROMPT = """
Extrahiere Daten aus folgendem radiologischem Befund im JSON Format:
"""


class ErrorCode(Enum):
    NO_ERROR = 0
    MISSING_FIELD = 1
    INVALID_VALUE = 2


class InputValidation(TypedDict):
    ecg_sync: ErrorCode
    density_tr_pulmonalis: ErrorCode
    artefact_score: ErrorCode
    lae_presence: ErrorCode
    lae_main_branch_right: ErrorCode
    lae_upper_lobe_right: ErrorCode
    lae_lower_lobe_right: ErrorCode
    lae_middle_lobe_right: ErrorCode
    lae_main_branch_left: ErrorCode
    lae_upper_lobe_left: ErrorCode
    lae_lower_lobe_left: ErrorCode
    clot_burden_score: ErrorCode
    perfusion_deficit: ErrorCode
    rv_lv_quotient: ErrorCode


class ClinicalInformation(BaseModel):
    keywords: list[str] = Field(
        ...,
        description="Was sind medizinisch relevante Schlagworte welche die klinischen Angaben oder "
        "die Fragestellung beschreiben? Fasse möglichst Begriffe zu Erkrankungsgruppen zusammen.",
    )
    morbidity: int = Field(
        ...,
        description="Entscheide anhand der klinischen Angaben die Morbidität des Patienten. "
        "Nutze dazu die Likert-Skala: "
        "1 = Neben der Frage nach LAE kein Hinweis auf eine Grunderkrankung, "
        "2 = Leichte Erkrankungslast, "
        "3 = Mittelschwere Erkrankungslast, "
        "4 = Schwere Erkrankungslast, "
        "5 = Sehr schwere Erkrankungslast",
    )
    symptom_duration: int | None = Field(
        ...,
        description="Wie lange bestehen die Symptome bereits (in Stunden)? "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    deep_vein_thrombosis: bool = Field(
        ..., description="Wurde eine Tiefe Beinvenenthrombose (TVT) angegeben?"
    )
    dyspnea: bool = Field(..., description="Ist eine Dyspnoe angegeben?")
    tachycardia: bool = Field(..., description="Ist eine Tachykardie angegeben?")
    pO2_reduction: bool = Field(..., description="Ist eine pO2-Reduktion angegeben?")
    pO2_percentage: int | None = Field(
        ...,
        description="Git es Informationen zum pO2-Wert? "
        "Falls ja, bitte den Wert eintragen. "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    troponin_elevated: bool = Field(
        ..., description="Ist ein erhöhter Troponin-Wert (Trop/TNT) angegeben?"
    )
    troponin_value: float | None = Field(
        ...,
        description="Ist ein Troponin-Wert (Trop/TNT) angegeben? "
        "Falls ja, bitte den Wert eintragen. "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    nt_pro_bnp_elevated: bool | None = Field(
        ..., description="Ist ein erhöhter NT-proBNP-Wert angegeben? "
    )
    nt_pro_bnp_value: float | None = Field(
        ...,
        description="Ist ein NT-proBNP-Wert angegeben? "
        "Falls ja, bitte den Wert eintragen. "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    d_dimers_elevated: bool = Field(..., description="Sind D-Dimere erhöht angegeben?")
    d_dimers_value: float | None = Field(
        ...,
        description="Sind D-Dimere angegeben? "
        "Falls ja, bitte den Wert eintragen. "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )


class Indication(BaseModel):
    inflammation_question: bool = Field(
        ...,
        description="Gibt es eine Frage nach entzündlicher Lungenerkrankung in der Fragestellung?",
    )
    lung_question: bool = Field(
        ...,
        description="Gibt es eine Frage nach einer anderen Lungenpathologie außer LAE oder "
        "Entzündung in der Fragestellung?",
    )
    aorta_question: bool = Field(
        ...,
        description="Gibt es eine Frage nach Erkrankung der Aorta in der Fragestellung?",
    )
    cardiac_question: bool = Field(
        ...,
        description="Gibt es eine Frage nach Erkrankung des Herzens in der Fragestellung?",
    )
    triple_rule_out_question: bool = Field(
        ...,
        description="Gibt es eine Frage nach Triple-Rule-Out in der Fragestellung?",
    )


class PerfusionDeficit(str, Enum):
    NONE = "Kein Perfusionsausfall"
    LT_25 = "< 25%"
    GE_25 = ">= 25%"


class LaePresence(str, Enum):
    NOT_ASSESSABLE = "Nicht beurteilbar"
    SUSPECTED = "Verdacht auf LAE"
    NO = "Nein"
    YES = "Ja"


class MainBranchOcclusion(str, Enum):
    NONE = "Keine Okklusion"
    TOTAL = "Totale Okklusion"
    PARTIAL = "Partielle Okklusion"


class LobeOcclusion(str, Enum):
    NONE = "Keine Okklusion"
    TOTAL = "Totale Okklusion"
    PARTIAL = "Partielle Okklusion"
    SEGMENTAL = "Segmentale Okklusion"
    SUBSEGMENTAL = "Subsegmentale Okklusion"


class Findings(BaseModel):
    ecg_sync: bool = Field(
        ..., description="Wurde das CT mit EKG-Synchronisation durchgeführt?"
    )
    density_tr_pulmonalis: int | None = Field(
        ...,
        description="Wie ist der angegebene Dichtegrad des Truncus pulmonalis? "
        "Angabe in Hounsfield Einheiten (HE/HU). "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    artefact_score: int | None = Field(
        ...,
        description="Was ist der angegebene Grad von Bewegungsartefakten? "
        "Falls keine Angabe dazu vorhanden ist, dann mit None antworten.",
    )
    previous_examination: bool = Field(
        ..., description="Ist eine Voraufnahme zum Vergleich angegeben?"
    )
    lae_presence: LaePresence = Field(
        ..., description="Wurde eine Lungenarterienembolie (LAE) gefunden?"
    )
    lae_main_branch_right: MainBranchOcclusion = Field(
        ..., description="Wurde eine LAE im Hauptstamm rechts gefunden?"
    )
    lae_upper_lobe_right: LobeOcclusion = Field(
        ..., description="Wurde eine LAE im Oberlappen rechts gefunden?"
    )
    lae_lower_lobe_right: LobeOcclusion = Field(
        ..., description="Wurde eine LAE im Unterlappen rechts gefunden?"
    )
    lae_middle_lobe_right: LobeOcclusion = Field(
        ..., description="Wurde eine LAE im Mittellappen rechts gefunden?"
    )
    lae_main_branch_left: MainBranchOcclusion = Field(
        ..., description="Wurde eine LAE im Hauptstamm links gefunden?"
    )
    lae_upper_lobe_left: LobeOcclusion = Field(
        ..., description="Wurde eine LAE im Oberlappen links gefunden?"
    )
    lae_lower_lobe_left: LobeOcclusion = Field(
        ..., description="Wurde eine LAE im Unterlappen links gefunden?"
    )
    # Valid values 0-40
    clot_burden_score: float | None = Field(
        ...,
        description="Wie hoch ist der Thrombuslastgrad beschrieben "
        "(Heidelberg Clot Burden Score)? "
        "Falls keine Angabe dazu vorhanden ist, dann mit None antworten.",
    )
    perfusion_deficit: PerfusionDeficit = Field(
        ...,
        description="Welche Perfusionsausfälle (DE-CT) sind beschrieben? "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    rv_lv_quotient: bool | None = Field(
        ...,
        description="Ist RV/LV-Quotient >= 1? "
        "Falls keine Angabe dazu vorhanden ist, dann mit 'None' antworten.",
    )
    inflammation: bool = Field(
        ..., description="Ist eine Entzündung im Befund beschrieben?"
    )
    congestion: bool = Field(..., description="Ist eine Stauung im Befund beschrieben?")
    suspect_finding: bool = Field(
        ..., description="Ist eine suspekte Läsion oder Tumor im Befund beschrieben?"
    )
    heart_pathology: bool = Field(
        ..., description="Ist eine Herzerkrankung im Befund beschrieben?"
    )
    vascular_pathology: bool = Field(
        ..., description="Ist eine Gefäßerkrankung im Befund beschrieben?"
    )
    bone_pathology: bool = Field(
        ..., description="Ist eine Knochenpathologie im Befund beschrieben?"
    )


class ExtractedData(BaseModel):
    clinical_information: ClinicalInformation = Field(
        ..., description="Daten extrahiert aus den klinischen Angaben."
    )
    indication: Indication = Field(
        ..., description="Daten extrahiert aus der Fragestellung."
    )
    findings: Findings = Field(..., description="Daten extrahiert aus dem Befund.")


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


class DataAnalyzer:
    def __init__(
        self,
        study_ids: list[str] | None,
        limit_reports: int | None,
        open_ai_base_url: str | None,
        open_ai_model: str,
        log_file: str,
        reports_file: str,
        validations_file: str,
        extracted_data_file: str,
        study_id_column: str,
        report_column: str,
    ) -> None:
        self.study_ids = study_ids
        self.limit_reports = limit_reports
        self.open_ai_base_url = open_ai_base_url
        self.open_ai_model = open_ai_model
        self.log_file = log_file
        self.reports_file = reports_file
        self.validations_file = validations_file
        self.extracted_data_file = extracted_data_file
        self.study_id_column = study_id_column
        self.report_column = report_column

        self.client = OpenAI(base_url=self.open_ai_base_url)

    def analyze(self) -> None:
        logging.info("Starting data analysis.")
        logging.info(f"Using LLM model: {self.open_ai_model}")

        logging.info("Loading data from CSV file.")
        df = pd.read_csv(self.reports_file)
        if self.study_ids is not None:
            df = pd.DataFrame(df[df[self.study_id_column].isin(self.study_ids)])
        elif isinstance(self.limit_reports, int):
            df = df.head(self.limit_reports)

        validations = self.validate_reports(df)
        self.export_validations(validations)

        extracted_data = self.extract_reports(df)
        self.export_extracted_data(extracted_data)

    def validate_reports(self, df: pd.DataFrame) -> dict[str, InputValidation]:
        logging.info("Validating inputs of reports.")

        validations: dict[str, InputValidation] = {}
        for _, row in track(
            df.iterrows(),
            total=df.shape[0],
            description="Validating input of reports ...",
        ):
            logging.info(f"Validating report of study ID: {row[self.study_id_column]}")

            report: str = str(row[self.report_column])
            study_id: str = str(row[self.study_id_column])
            validation = self.validate_report(report, study_id)
            validations[study_id] = validation

        return validations

    def validate_report(self, report: str, study_id: str) -> InputValidation:
        """Validate the report with below options."""

        validation = InputValidation(
            ecg_sync=ErrorCode.NO_ERROR,
            density_tr_pulmonalis=ErrorCode.NO_ERROR,
            artefact_score=ErrorCode.NO_ERROR,
            lae_presence=ErrorCode.NO_ERROR,
            lae_main_branch_right=ErrorCode.NO_ERROR,
            lae_upper_lobe_right=ErrorCode.NO_ERROR,
            lae_lower_lobe_right=ErrorCode.NO_ERROR,
            lae_middle_lobe_right=ErrorCode.NO_ERROR,
            lae_main_branch_left=ErrorCode.NO_ERROR,
            lae_upper_lobe_left=ErrorCode.NO_ERROR,
            lae_lower_lobe_left=ErrorCode.NO_ERROR,
            clot_burden_score=ErrorCode.NO_ERROR,
            perfusion_deficit=ErrorCode.NO_ERROR,
            rv_lv_quotient=ErrorCode.NO_ERROR,
        )

        value = self.get_field_value(report, "EKG-Synchronisation")
        if value is None:
            validation["ecg_sync"] = ErrorCode.MISSING_FIELD
        elif value not in ["", "Nein", "Ja"]:
            logging.error(f"Invalid value for 'EKG-Synchronisation': {value}")
            validation["ecg_sync"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "CT-Dichte Truncus pulmonalis (Standard)")
        if value is None:
            validation["density_tr_pulmonalis"] = ErrorCode.MISSING_FIELD
        elif value != "-" and not re.fullmatch(r"\d+(,\d+)? HU", value):
            logging.error(
                f"Invalid value for 'CT-Dichte Truncus pulmonalis (Standard)': {value}"
            )
            validation["density_tr_pulmonalis"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "Artefakt-Score (0-5)")
        if value is None:
            validation["artefact_score"] = ErrorCode.MISSING_FIELD
        elif value not in [
            "",
            "0 (keine Artefakte)",
            "1",
            "2",
            "3",
            "4",
            "5 (nicht beurteilbar)",
        ]:
            logging.error(f"Invalid value for 'Artefakt-Score (0-5)': {value}")
            validation["artefact_score"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "Nachweis einer Lungenarterienembolie")
        if value is None:
            validation["lae_presence"] = ErrorCode.MISSING_FIELD
        elif value not in [
            "Nein",
            "Ja",
            "Verdacht auf",
            "Nicht beurteilbar",
        ]:
            logging.error(
                f"Invalid value for 'Nachweis einer Lungenarterienembolie': {value}"
            )
            validation["lae_presence"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(
            report, "Heidelberg Clot Burden Score (CBS, PMID: 34581626)"
        )
        if value is None:
            validation["clot_burden_score"] = ErrorCode.MISSING_FIELD
        elif not re.fullmatch(r"\d+(,\d+)?", value) or not (
            0 <= float(value.replace(",", ".")) <= 40
        ):
            logging.error(
                f"Invalid value for 'Heidelberg Clot Burden Score (CBS, PMID: 34581626)': {value}"
            )
            validation["clot_burden_score"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "Perfusionsausfälle (DE-CT)")
        if value is None:
            validation["perfusion_deficit"] = ErrorCode.MISSING_FIELD
        elif value not in ["-", "Keine", "<25%", "≥25%", "=25%"]:
            logging.error(f"Invalid value for 'Perfusionsausfälle (DE-CT)': {value}")
            validation["perfusion_deficit"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "RV/LV-Quotient")
        if value is None:
            validation["rv_lv_quotient"] = ErrorCode.MISSING_FIELD
        elif value not in ["-", "<1", "≥1", "=1"]:
            logging.error(f"Invalid value for 'RV/LV-Quotient': {value}")
            validation["rv_lv_quotient"] = ErrorCode.INVALID_VALUE

        main_branches = {
            "Rechts Pulmonalhauptarterie": "lae_main_branch_right",
            "Links Pulmonalhauptarterie": "lae_main_branch_left",
        }
        for main_branch in main_branches:
            value = self.get_field_value(report, main_branch)
            if value is None:
                validation[main_branches[main_branch]] = ErrorCode.MISSING_FIELD
            elif value not in [
                "",
                "-",
                "Total okkludiert",
                "Partiell okkludiert",
            ]:
                logging.error(f"Invalid value for '{main_branch}': {value}")
                validation[main_branches[main_branch]] = ErrorCode.INVALID_VALUE

        lobes = {
            "Rechts Oberlappen": "lae_upper_lobe_right",
            "Rechts Unterlappen": "lae_lower_lobe_right",
            "Mittellappen": "lae_middle_lobe_right",
            "Links Oberlappen": "lae_upper_lobe_left",
            "Links Unterlappen": "lae_lower_lobe_left",
        }
        for lobe in lobes:
            value = self.get_field_value(report, lobe)
            if value is None:
                validation[lobes[lobe]] = ErrorCode.MISSING_FIELD
            elif value not in [
                "",
                "-",
                "Lappenarterie total okkludiert",
                "Lappenarterie partiell okkludiert",
                "Segmentarterie(n)",
                "Subsegmentarterie(n)",
            ]:
                logging.error(f"Invalid value for '{lobe}': {value}")
                validation[lobes[lobe]] = ErrorCode.INVALID_VALUE

        # If all main branches and lobes where missing then this is not an error
        # and set all to NO_ERROR
        if all(
            validation[main_branch] == ErrorCode.MISSING_FIELD
            for main_branch in main_branches.values()
        ) and all(
            validation[lobe] == ErrorCode.MISSING_FIELD for lobe in lobes.values()
        ):
            for main_branch in main_branches.values():
                validation[main_branch] = ErrorCode.NO_ERROR
            for lobe in lobes.values():
                validation[lobe] = ErrorCode.NO_ERROR

        return validation

    def get_field_value(self, report: str, field: str) -> str | None:
        for line in report.split("\n"):
            if f"{field}:" in line:
                return line.split(":")[-1].strip()

        logging.error(f"No field found with name: '{field}'")

    def export_validations(self, validations: dict[str, InputValidation]) -> None:
        logging.info(
            f"Exporting input validations to CSV file '{self.validations_file}'."
        )

        items: list[dict[str, Any]] = []
        for study_id, validation in validations.items():
            item: dict[str, Any] = {"study_id": study_id}
            item = item | validation
            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.validations_file, index=False)

    def extract_reports(self, df: pd.DataFrame) -> dict[str, ExtractedData]:
        logging.info("Extracting data from reports.")

        extracted_data: dict[str, ExtractedData] = {}
        for _, row in track(
            df.iterrows(),
            total=df.shape[0],
            description="Extracting data from reports ...",
        ):
            logging.info(f"Extracting data of study ID: {row[self.study_id_column]}")

            report: str = str(row[self.report_column])
            study_id: str = str(row[self.study_id_column])
            data = self.extract_report(report)
            extracted_data[study_id] = data

        return extracted_data

    def extract_report(self, report: str) -> ExtractedData:
        completion = self.client.beta.chat.completions.parse(
            model=self.open_ai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": report},
            ],
            response_format=ExtractedData,
        )

        extracted_data = completion.choices[0].message.parsed
        assert extracted_data
        return extracted_data

    def export_extracted_data(self, extracted_data: dict[str, ExtractedData]) -> None:
        logging.info(
            f"Exporting extracted data to CSV file '{self.extracted_data_file}'."
        )

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
            item["clot_burden_score_calc"] = self.calc_cbs_score(data.findings)

            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.extracted_data_file, index=False)

    def calc_cbs_score(self, findings: Findings) -> float:
        score: float = 0

        # Left lung
        if findings.lae_main_branch_left == MainBranchOcclusion.TOTAL:
            score += 20
        elif findings.lae_main_branch_left == MainBranchOcclusion.PARTIAL:
            score += 10
        else:
            if findings.lae_upper_lobe_left == LobeOcclusion.TOTAL:
                score += 10
            elif findings.lae_upper_lobe_left == LobeOcclusion.PARTIAL:
                score += 5
            elif findings.lae_upper_lobe_left == LobeOcclusion.SEGMENTAL:
                score += 2.5
            elif findings.lae_upper_lobe_left == LobeOcclusion.SUBSEGMENTAL:
                score += 1

            if findings.lae_lower_lobe_left == LobeOcclusion.TOTAL:
                score += 10
            elif findings.lae_lower_lobe_left == LobeOcclusion.PARTIAL:
                score += 5
            elif findings.lae_lower_lobe_left == LobeOcclusion.SEGMENTAL:
                score += 2.5
            elif findings.lae_lower_lobe_left == LobeOcclusion.SUBSEGMENTAL:
                score += 1

        # Right lung
        if findings.lae_main_branch_right == MainBranchOcclusion.TOTAL:
            score += 20
        elif findings.lae_main_branch_right == MainBranchOcclusion.PARTIAL:
            score += 10
        else:
            if findings.lae_upper_lobe_right == LobeOcclusion.TOTAL:
                score += 6
            elif findings.lae_upper_lobe_right == LobeOcclusion.PARTIAL:
                score += 3
            elif findings.lae_upper_lobe_right == LobeOcclusion.SEGMENTAL:
                score += 1.5
            elif findings.lae_upper_lobe_right == LobeOcclusion.SUBSEGMENTAL:
                score += 1

            if findings.lae_middle_lobe_right == LobeOcclusion.TOTAL:
                score += 4
            elif findings.lae_middle_lobe_right == LobeOcclusion.PARTIAL:
                score += 2
            elif findings.lae_middle_lobe_right == LobeOcclusion.SEGMENTAL:
                score += 1
            elif findings.lae_middle_lobe_right == LobeOcclusion.SUBSEGMENTAL:
                score += 0.5

            if findings.lae_lower_lobe_right == LobeOcclusion.TOTAL:
                score += 10
            elif findings.lae_lower_lobe_right == LobeOcclusion.PARTIAL:
                score += 5
            elif findings.lae_lower_lobe_right == LobeOcclusion.SEGMENTAL:
                score += 2.5
            elif findings.lae_lower_lobe_right == LobeOcclusion.SUBSEGMENTAL:
                score += 1

        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int)
    parser.add_argument("-s", "--study-id", action="append")
    args = parser.parse_args()

    limit_reports: int | None = None
    if args.limit is not None:
        limit_reports = args.limit

    study_ids: list[str] | None = None
    if args.study_id:
        study_ids = args.study_id

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

    analyzer = DataAnalyzer(
        limit_reports=limit_reports,
        study_ids=study_ids,
        open_ai_base_url=open_ai_base_url,
        open_ai_model=open_ai_model,
        log_file=log_file,
        reports_file=reports_file,
        validations_file=validations_file,
        extracted_data_file=extracted_data_file,
        study_id_column=study_id_column,
        report_column=report_column,
    )
    analyzer.analyze()


if __name__ == "__main__":
    main()
