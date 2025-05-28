import logging
import re
from typing import Any

import pandas as pd
from rich.progress import track

from llm_lae.llm_models import (
    LaePresence,
    LobeOcclusion,
    MainBranchOcclusion,
    PerfusionDeficit,
    RightHeartQuotient,
)

from .generic_models import Report
from .rules_models import ErrorCode, EvaluatedValues, InputValues, RulesResult


class RulesExtractor:
    def __init__(
        self,
        reports: list[Report],
        input_values_file: str,
        evaluated_values_file: str,
    ):
        self.reports = reports
        self.input_values_file = input_values_file
        self.evaluated_values_file = evaluated_values_file

    def extract(self):
        logging.info("Validating inputs of reports.")

        results = self.extract_from_reports()
        self.export_extracted_input_values(results)
        self.export_extracted_evaluated_values(results)

    def extract_from_reports(self) -> list[RulesResult]:
        results: list[RulesResult] = []
        for report in track(
            self.reports,
            description="Validating input of reports ...",
        ):
            logging.info(f"Validating report of study ID: {report['study_id']}")

            result = self.extract_from_report(report["report_body"], report["study_id"])
            results.append(result)

        return results

    def extract_from_report(self, report: str, study_id: str) -> RulesResult:
        """Validate the report with below options."""

        ecg_sync_iv, ecg_sync_ev = self.extract_ecg_sync(report)
        density_tr_pulmonalis_iv, density_tr_pulmonalis_ev = self.extract_density_tr_pulmonalis(
            report
        )
        artefact_score_iv, artefact_score_ev = self.extract_artefact_score(report)
        lae_presence_iv, lae_presence_ev = self.extract_lae_presence(report)
        lae_main_branch_right_iv, lae_main_branch_right_ev = self.extract_main_branch_occlusion(
            report, "Rechts Pulmonalhauptarterie"
        )
        lae_upper_lobe_right_iv, lae_upper_lobe_right_ev = self.extract_lobe_occlusion(
            report, "Rechts Oberlappen"
        )
        lae_lower_lobe_right_iv, lae_lower_lobe_right_ev = self.extract_lobe_occlusion(
            report, "Rechts Unterlappen"
        )
        lae_middle_lobe_right_iv, lae_middle_lobe_right_ev = self.extract_lobe_occlusion(
            report, "Mittellappen"
        )
        lae_main_branch_left_iv, lae_main_branch_left_ev = self.extract_main_branch_occlusion(
            report, "Links Pulmonalhauptarterie"
        )
        lae_upper_lobe_left_iv, lae_upper_lobe_left_ev = self.extract_lobe_occlusion(
            report, "Links Oberlappen"
        )
        lae_lower_lobe_left_iv, lae_lower_lobe_left_ev = self.extract_lobe_occlusion(
            report, "Links Unterlappen"
        )
        clot_burden_score_iv, clot_burden_score_ev = self.extract_clot_burden_score(report)
        perfusion_deficit_iv, perfusion_deficit_ev = self.extract_perfusion_deficit(report)
        rv_lv_quotient_iv, rv_lv_quotient_ev = self.extract_rv_lv_quotient(report)

        # If all main branches and lobes are missing then this is not an error
        # but means that nothing is occluded.
        if all(
            value == ErrorCode.MISSING_FIELD
            for value in [
                lae_main_branch_right_ev,
                lae_upper_lobe_right_ev,
                lae_lower_lobe_right_ev,
                lae_middle_lobe_right_ev,
                lae_main_branch_left_ev,
                lae_upper_lobe_left_ev,
                lae_lower_lobe_left_ev,
            ]
        ):
            lae_main_branch_right_ev = MainBranchOcclusion.NONE
            lae_upper_lobe_right_ev = LobeOcclusion.NONE
            lae_lower_lobe_right_ev = LobeOcclusion.NONE
            lae_middle_lobe_right_ev = LobeOcclusion.NONE
            lae_main_branch_left_ev = MainBranchOcclusion.NONE
            lae_upper_lobe_left_ev = LobeOcclusion.NONE
            lae_lower_lobe_left_ev = LobeOcclusion.NONE

        input_values = InputValues(
            ecg_sync=ecg_sync_iv,
            density_tr_pulmonalis=density_tr_pulmonalis_iv,
            artefact_score=artefact_score_iv,
            lae_presence=lae_presence_iv,
            lae_main_branch_right=lae_main_branch_right_iv,
            lae_upper_lobe_right=lae_upper_lobe_right_iv,
            lae_lower_lobe_right=lae_lower_lobe_right_iv,
            lae_middle_lobe_right=lae_middle_lobe_right_iv,
            lae_main_branch_left=lae_main_branch_left_iv,
            lae_upper_lobe_left=lae_upper_lobe_left_iv,
            lae_lower_lobe_left=lae_lower_lobe_left_iv,
            clot_burden_score=clot_burden_score_iv,
            perfusion_deficit=perfusion_deficit_iv,
            rv_lv_quotient=rv_lv_quotient_iv,
        )

        evaluated_values = EvaluatedValues(
            ecg_sync=ecg_sync_ev,
            density_tr_pulmonalis=density_tr_pulmonalis_ev,
            artefact_score=artefact_score_ev,
            lae_presence=lae_presence_ev,
            lae_main_branch_right=lae_main_branch_right_ev,
            lae_upper_lobe_right=lae_upper_lobe_right_ev,
            lae_lower_lobe_right=lae_lower_lobe_right_ev,
            lae_middle_lobe_right=lae_middle_lobe_right_ev,
            lae_main_branch_left=lae_main_branch_left_ev,
            lae_upper_lobe_left=lae_upper_lobe_left_ev,
            lae_lower_lobe_left=lae_lower_lobe_left_ev,
            clot_burden_score=clot_burden_score_ev,
            perfusion_deficit=perfusion_deficit_ev,
            rv_lv_quotient=rv_lv_quotient_ev,
        )

        return RulesResult(
            study_id=study_id,
            input_values=input_values,
            evaluated_values=evaluated_values,
        )

    def extract_ecg_sync(self, report: str) -> tuple[str, bool | ErrorCode]:
        ev: bool | ErrorCode
        iv = self.get_field_value(report, "EKG-Synchronisation")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv == "Ja":
            ev = True
        elif iv in ["Nein", "", "-"]:
            ev = False
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_density_tr_pulmonalis(self, report: str) -> tuple[str, int | None | ErrorCode]:
        ev: int | None | ErrorCode
        iv = self.get_field_value(report, "CT-Dichte Truncus pulmonalis (Standard)")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = None
        elif re.fullmatch(r"\d+(,\d+)? HU", iv):
            ev = round(float(iv.split(" ")[0].replace(",", ".")))
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_artefact_score(self, report: str) -> tuple[str, int | None | ErrorCode]:
        ev: int | None | ErrorCode
        iv = self.get_field_value(report, "Artefakt-Score (0-5)")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = None
        elif iv in ["0 (keine Artefakte)", "1", "2", "3", "4", "5 (nicht beurteilbar)"]:
            ev = int(iv[0])
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_lae_presence(self, report: str) -> tuple[str, LaePresence | ErrorCode]:
        ev: LaePresence | ErrorCode
        iv = self.get_field_value(report, "Nachweis einer Lungenarterienembolie")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv == "Nein":
            ev = LaePresence.NO
        elif iv == "Ja":
            ev = LaePresence.YES
        elif iv == "Verdacht auf Lungenarterienembolie":
            ev = LaePresence.SUSPECTED
        elif iv == "Nicht beurteilbar":
            ev = LaePresence.NOT_ASSESSABLE
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_clot_burden_score(self, report: str) -> tuple[str, float | None | ErrorCode]:
        ev: float | None | ErrorCode
        iv = self.get_field_value(report, "Heidelberg Clot Burden Score (CBS, PMID: 34581626)")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif not re.fullmatch(r"\d+(,\d+)?", iv) or not (0 <= float(iv.replace(",", ".")) <= 40):
            ev = ErrorCode.INVALID_VALUE
        else:
            ev = float(iv.replace(",", "."))

        return iv, ev

    def extract_perfusion_deficit(
        self, report: str
    ) -> tuple[str, PerfusionDeficit | None | ErrorCode]:
        ev: PerfusionDeficit | None | ErrorCode
        iv = self.get_field_value(report, "Perfusionsausfälle (DE-CT)")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = None
        elif iv == "Keine":
            ev = PerfusionDeficit.NONE
        elif iv == "<25%":
            ev = PerfusionDeficit.LT_25
        elif iv in ["≥25%", "=25%"]:
            ev = PerfusionDeficit.GE_25
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_rv_lv_quotient(
        self, report: str
    ) -> tuple[str, RightHeartQuotient | None | ErrorCode]:
        ev: RightHeartQuotient | None | ErrorCode
        iv = self.get_field_value(report, "RV/LV-Quotient")
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = None
        elif iv == "<1":
            ev = RightHeartQuotient.LT_1
        elif iv in ["≥1", "=1"]:
            ev = RightHeartQuotient.GE_1
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_main_branch_occlusion(
        self, report: str, branch: str
    ) -> tuple[str, MainBranchOcclusion | ErrorCode]:
        ev: MainBranchOcclusion | ErrorCode
        iv = self.get_field_value(report, branch)
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = MainBranchOcclusion.NONE
        elif iv == "Total okkludiert":
            ev = MainBranchOcclusion.TOTAL
        elif iv == "Partiell okkludiert":
            ev = MainBranchOcclusion.PARTIAL
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def extract_lobe_occlusion(
        self, report: str, lobe: str
    ) -> tuple[str, LobeOcclusion | ErrorCode]:
        ev: LobeOcclusion | ErrorCode
        iv = self.get_field_value(report, lobe)
        if iv is None:
            iv = ""
            ev = ErrorCode.MISSING_FIELD
        elif iv in ["", "-"]:
            ev = LobeOcclusion.NONE
        elif iv == "Lappenarterie total okkludiert":
            ev = LobeOcclusion.TOTAL
        elif iv == "Lappenarterie partiell okkludiert":
            ev = LobeOcclusion.PARTIAL
        elif iv == "Segmentarterie(n)":
            ev = LobeOcclusion.SEGMENTAL
        elif iv == "Subsegmentarterie(n)":
            ev = LobeOcclusion.SUBSEGMENTAL
        else:
            ev = ErrorCode.INVALID_VALUE

        return iv, ev

    def get_field_value(self, report: str, field: str) -> str | None:
        for line in report.split("\n"):
            if f"{field}:" in line:
                return line.split(":")[-1].strip()

        logging.error(f"No field found with name: '{field}'")

    def export_extracted_input_values(self, results: list[RulesResult]) -> None:
        logging.info(f"Exporting input values to '{self.input_values_file}'.")

        # Export input values
        items: list[dict[str, Any]] = []
        for result in results:
            item: dict[str, Any] = {"study_id": result.study_id}
            item = item | result.input_values.model_dump()
            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.input_values_file, index=False)

    def export_extracted_evaluated_values(self, results: list[RulesResult]) -> None:
        logging.info(f"Exporting evaluated values to '{self.input_values_file}'.")

        items: list[dict[str, Any]] = []
        for result in results:
            item: dict[str, Any] = {"study_id": result.study_id}
            item = item | result.evaluated_values.model_dump()
            items.append(item)

        df = pd.DataFrame(items)
        df.to_csv(self.evaluated_values_file, index=False)
