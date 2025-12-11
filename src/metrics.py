from __future__ import annotations

from typing import Iterable, List, Tuple

import Levenshtein


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def cer(ref: str, hyp: str) -> float:
    """
    Character Error Rate = Levenshtein(ref, hyp) / len(ref).
    """
    ref_norm = _normalize_text(ref)
    hyp_norm = _normalize_text(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = Levenshtein.distance(ref_norm, hyp_norm)
    return dist / len(ref_norm)


def wer(ref: str, hyp: str) -> float:
    """
    Word Error Rate по словам, разделённым пробелами.
    """
    ref_words = _normalize_text(ref).split()
    hyp_words = _normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Простейшая DP для расстояния Левенштейна по словам
    n, m = len(ref_words), len(hyp_words)
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m] / n


def batch_cer_wer(pairs: Iterable[Tuple[str, str]]) -> Tuple[float, float]:
    """
    Усреднённые CER и WER по набору пар (ref, hyp).
    """
    cer_vals: List[float] = []
    wer_vals: List[float] = []
    for ref, hyp in pairs:
        cer_vals.append(cer(ref, hyp))
        wer_vals.append(wer(ref, hyp))
    if not cer_vals:
        return 0.0, 0.0
    return sum(cer_vals) / len(cer_vals), sum(wer_vals) / len(wer_vals)
