from enum import Enum

from pydantic import BaseModel

from llm_lae.llm_models import (
    LaePresence,
    LobeOcclusion,
    MainBranchOcclusion,
    PerfusionDeficit,
    RightHeartQuotient,
)


class ErrorCode(str, Enum):
    MISSING_FIELD = "Missing field"
    INVALID_VALUE = "Invalid value"


class InputValues(BaseModel):
    ecg_sync: str
    density_tr_pulmonalis: str
    artefact_score: str
    lae_presence: str
    lae_main_branch_right: str
    lae_upper_lobe_right: str
    lae_lower_lobe_right: str
    lae_middle_lobe_right: str
    lae_main_branch_left: str
    lae_upper_lobe_left: str
    lae_lower_lobe_left: str
    clot_burden_score: str
    perfusion_deficit: str
    rv_lv_quotient: str


class EvaluatedValues(BaseModel):
    ecg_sync: bool | ErrorCode
    density_tr_pulmonalis: int | None | ErrorCode
    artefact_score: int | None | ErrorCode
    lae_presence: LaePresence | ErrorCode
    lae_main_branch_right: MainBranchOcclusion | ErrorCode
    lae_upper_lobe_right: LobeOcclusion | ErrorCode
    lae_lower_lobe_right: LobeOcclusion | ErrorCode
    lae_middle_lobe_right: LobeOcclusion | ErrorCode
    lae_main_branch_left: MainBranchOcclusion | ErrorCode
    lae_upper_lobe_left: LobeOcclusion | ErrorCode
    lae_lower_lobe_left: LobeOcclusion | ErrorCode
    clot_burden_score: float | None | ErrorCode
    perfusion_deficit: PerfusionDeficit | None | ErrorCode
    rv_lv_quotient: RightHeartQuotient | None | ErrorCode


class RulesResult(BaseModel):
    study_id: str
    input_values: InputValues
    evaluated_values: EvaluatedValues
