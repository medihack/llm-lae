from enum import Enum
import os
from typing import Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from rich.progress import track
import pandas as pd
import re
import logging


load_dotenv()

SYSTEM_PROMPT: str = (
    "Extrahiere Daten aus folgendem strukturiertem radiologischem Befund:"
)


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
        open_ai_base_url: str | None,
        open_ai_model: str,
        limit_reports: int | None,
        log_file: str,
        input_file: str,
        output_file: str,
        study_id_column: str,
        report_column: str,
    ) -> None:
        self.open_ai_base_url = open_ai_base_url
        self.open_ai_model = open_ai_model
        self.limit_reports = limit_reports
        self.log_file = log_file
        self.input_file = input_file
        self.output_file = output_file
        self.study_id_column = study_id_column
        self.report_column = report_column

        self.client = OpenAI(base_url=self.open_ai_base_url)

    def analyze(self) -> None:
        logging.info("Starting data analysis.")
        logging.info(f"Using LLM model: {self.open_ai_model}")

        logging.info("Loading data from CSV file.")
        df = pd.read_csv(self.input_file)
        if self.limit_reports is not None:
            df = df.head(self.limit_reports)

        self.validate_reports(df, self.study_id_column, self.report_column)
        extracted_data = self.extract_reports(
            df, self.study_id_column, self.report_column
        )
        self.export_data(extracted_data, self.output_file)

    def validate_reports(
        self, df: pd.DataFrame, study_id_column: str, report_column: str
    ) -> None:
        logging.info("Validating inputs of reports.")

        for _, row in track(
            df.iterrows(),
            total=df.shape[0],
            description="Validating input of reports ...",
        ):
            logging.info(f"Validating report of study ID: {row[study_id_column]}")

            report: str = str(row[report_column])
            study_id: str = str(row[study_id_column])
            self.validate_report(report, study_id)

    def validate_report(self, report: str, study_id: str) -> None:
        """Validate the report with below options."""

        value = self.get_field_value(report, "EKG-Synchronisation")
        if value is not None and value not in ["", "Nein", "Ja"]:
            logging.error(f"Invalid value for 'EKG-Synchronisation': {value}")

        value = self.get_field_value(report, "CT-Dichte Truncus pulmonalis (Standard)")
        if (
            value is not None
            and value != "-"
            and not re.fullmatch(r"\d+(,\d+)? HU", value)
        ):
            logging.error(
                f"Invalid value for 'CT-Dichte Truncus pulmonalis (Standard)': {value}"
            )

        value = self.get_field_value(report, "Artefakt-Score (0-5)")
        if value is not None and value not in [
            "",
            "0 (keine Artefakte)",
            "1",
            "2",
            "3",
            "4",
            "5 (nicht beurteilbar)",
        ]:
            logging.error(f"Invalid value for 'Artefakt-Score (0-5)': {value}")

        value = self.get_field_value(report, "Nachweis einer Lungenarterienembolie")
        if value is not None and value not in [
            "Nein",
            "Ja",
            "Verdacht auf",
            "Nicht beurteilbar",
        ]:
            logging.error(
                f"Invalid value for 'Nachweis einer Lungenarterienembolie': {value}"
            )

        value = self.get_field_value(
            report, "Heidelberg Clot Burden Score (CBS, PMID: 34581626)"
        )
        if value is not None and (
            not re.fullmatch(r"\d+(,\d+)?", value)
            or not (0 <= float(value.replace(",", ".")) <= 40)
        ):
            logging.error(
                f"Invalid value for 'Heidelberg Clot Burden Score (CBS, PMID: 34581626)': {value}"
            )

        value = self.get_field_value(report, "Perfusionsausfälle (DE-CT)")
        if value is not None and value not in ["-", "Keine", "<25%", "≥25%"]:
            logging.error(f"Invalid value for 'Perfusionsausfälle (DE-CT)': {value}")

        value = self.get_field_value(report, "RV/LV-Quotient")
        if value is not None and value not in ["-", "<1", "≥1"]:
            logging.error(f"Invalid value for 'RV/LV-Quotient': {value}")

        for main_branch in [
            "Rechts Pulmonalhauptarterie",
            "Links Pulmonalhauptarterie",
        ]:
            value = self.get_field_value(report, main_branch)
            if value is not None and value not in [
                "",
                "-",
                "Total okkludiert",
                "Partiell okkludiert",
            ]:
                logging.error(f"Invalid value for '{main_branch}': {value}")

        for lobe in [
            "Rechts Oberlappen",
            "Mittellappen",
            "Rechts Unterlappen",
            "Links Oberlappen",
            "Links Unterlappen",
        ]:
            value = self.get_field_value(report, lobe)
            if value is not None and value not in [
                "",
                "-",
                "Lappenarterie total okkludiert",
                "Lappenarterie partiell okkludiert",
                "Segmentarterie(n)",
                "Subsegmentarterie(n)",
            ]:
                logging.error(f"Invalid value for '{lobe}': {value}")

    def get_field_value(self, report: str, field: str) -> str | None:
        for line in report.split("\n"):
            if field in line:
                return line.split(":")[-1].strip()

        logging.error(f"No field found with name: '{field}'")

    def extract_reports(
        self, df: pd.DataFrame, study_id_column: str, report_column: str
    ) -> dict[str, ExtractedData]:
        logging.info("Extracting data from reports.")

        extracted_data: dict[str, ExtractedData] = {}
        for _, row in track(
            df.iterrows(),
            total=df.shape[0],
            description="Extracting data from reports ...",
        ):
            logging.info(f"Extracting data of study ID: {row[study_id_column]}")

            report: str = str(row[report_column])
            study_id: str = str(row[study_id_column])
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

    def export_data(
        self, extracted_data: dict[str, ExtractedData], output_file: str
    ) -> None:
        logging.info("Exporting data to SQLite database.")

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
        df.to_csv(output_file, index=False)

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
    open_ai_base_url = os.getenv("OPENAI_BASE_URL")

    open_ai_model = os.getenv("OPENAI_MODEL")
    if not open_ai_model:
        raise ValueError("OPENAI_MODEL environment variable is not set.")

    limit_reports: int | None = None
    if limit := os.getenv("LIMIT_REPORTS"):
        limit_reports = int(limit)

    log_file = os.getenv("LOG_FILE")
    if not log_file:
        raise ValueError("LOG_FILE environment variable is not set.")

    input_file = os.getenv("INPUT_FILE")
    if not input_file:
        raise ValueError("INPUT_FILE environment variable is not set.")

    output_file = os.getenv("OUTPUT_FILE")
    if not output_file:
        raise ValueError("OUTPUT_FILE environment variable is not set.")

    study_id_column = os.getenv("STUDY_ID_COLUMN")
    if not study_id_column:
        raise ValueError("STUDY_ID_COLUMN environment variable is not set.")

    report_column = os.getenv("REPORT_COLUMN")
    if not report_column:
        raise ValueError("REPORT_COLUMN environment variable is not set.")

    setup_logging(log_file)

    analyzer = DataAnalyzer(
        open_ai_base_url=open_ai_base_url,
        open_ai_model=open_ai_model,
        limit_reports=limit_reports,
        log_file=log_file,
        input_file=input_file,
        output_file=output_file,
        study_id_column=study_id_column,
        report_column=report_column,
    )
    analyzer.analyze()


if __name__ == "__main__":
    main()
