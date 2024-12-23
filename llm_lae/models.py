from enum import Enum
from typing import TypedDict

from pydantic import BaseModel


class Report(TypedDict):
    study_id: str
    report_body: str


class ClinicalInformation(BaseModel):
    keywords: list[str]
    morbidity: int
    symptom_duration: int | None
    deep_vein_thrombosis: bool
    dyspnea: bool
    tachycardia: bool
    pO2_reduction: bool
    pO2_percentage: int | None
    troponin_elevated: bool
    troponin_value: int | None
    nt_pro_bnp_elevated: bool
    nt_pro_bnp_value: int | None
    d_dimers_elevated: bool
    d_dimers_value: float | None


class Indication(BaseModel):
    inflammation_question: bool
    lung_question: bool
    aorta_question: bool
    cardiac_question: bool
    triple_rule_out_question: bool


class PerfusionDeficit(str, Enum):
    NONE = "NA"
    NO_25 = "Keine"
    LT_25 = "< 25%"
    GE_25 = "≥ 25%"


class RightHeartQuotient(str, Enum):
    NONE = "NA"
    LT_1 = "< 1"
    GE_1 = "≥ 1"


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
    ecg_sync: bool
    density_tr_pulmonalis: int | None
    artefact_score: int | None
    previous_examination: bool
    lae_presence: LaePresence
    lae_main_branch_right: MainBranchOcclusion
    lae_upper_lobe_right: LobeOcclusion
    lae_lower_lobe_right: LobeOcclusion
    lae_middle_lobe_right: LobeOcclusion
    lae_main_branch_left: MainBranchOcclusion
    lae_upper_lobe_left: LobeOcclusion
    lae_lower_lobe_left: LobeOcclusion
    clot_burden_score: float | None
    perfusion_deficit: PerfusionDeficit
    rv_lv_quotient: RightHeartQuotient
    inflammation: bool
    congestion: bool
    suspect_finding: bool
    heart_pathology: bool
    vascular_pathology: bool
    bone_pathology: bool


class ExtractedData(BaseModel):
    clinical_information: ClinicalInformation
    indication: Indication
    findings: Findings


class LlmResult(TypedDict):
    extracted_data: ExtractedData
    study_id: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
