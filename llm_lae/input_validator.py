import logging
import re
from enum import Enum
from typing import Any, TypedDict

import pandas as pd
from rich.progress import track

from .models import Report


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


class InputValidator:
    def __init__(
        self,
        reports: list[Report],
        validations_file: str,
    ):
        self.reports = reports
        self.validations_file = validations_file

    def validate(self):
        logging.info("Validating inputs of reports.")

        validations = self.validate_reports()
        self.export_validations(validations)

    def validate_reports(self) -> dict[str, InputValidation]:
        validations: dict[str, InputValidation] = {}
        for report in track(
            self.reports,
            description="Validating input of reports ...",
        ):
            logging.info(f"Validating report of study ID: {report['study_id']}")

            validation = self.validate_report(report["report_body"], report["study_id"])
            validations[report["study_id"]] = validation

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
            logging.error(f"Invalid value for 'CT-Dichte Truncus pulmonalis (Standard)': {value}")
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
            logging.error(f"Invalid value for 'Nachweis einer Lungenarterienembolie': {value}")
            validation["lae_presence"] = ErrorCode.INVALID_VALUE

        value = self.get_field_value(report, "Heidelberg Clot Burden Score (CBS, PMID: 34581626)")
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
        ) and all(validation[lobe] == ErrorCode.MISSING_FIELD for lobe in lobes.values()):
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
        logging.info(f"Exporting input validations to CSV file '{self.validations_file}'.")

        items: list[dict[str, Any]] = []
        for study_id, validation in validations.items():
            item: dict[str, Any] = {"study_id": study_id}
            item = item | validation
            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.validations_file, index=False)
