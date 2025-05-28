import re

from .llm_models import Findings, LobeOcclusion, MainBranchOcclusion


def sanitize_filename(filename):
    # Remove invalid characters: \ / : * ? " < > |
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", filename)
    # Remove control characters (ASCII codes 0-31)
    sanitized = re.sub(r"[\x00-\x1f]", "_", sanitized)
    # Trim trailing spaces and periods
    sanitized = sanitized.rstrip(" .")
    return sanitized


def calc_cbs_score(findings: Findings) -> float:
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
