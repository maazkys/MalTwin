# modules/fusion/fusion.py
"""
Hybrid static + dynamic decision-level fusion.

Inputs
  static  : CNN result for a binary — predicted Malimg family and top-1
            confidence. NOTE: the CNN is trained on malware only (Malimg has no
            benign class), so its confidence means "how strongly this binary
            matches a KNOWN malware family", not "probability of malware".
  dynamic : DynamicReport for the traffic captured while the sample ran —
            capture_score is P(malicious behaviour).

Fused score
  fused = (w_s * static_conf + w_d * dynamic_score) / (w_s + w_d)
  using only the sources that are present. Weights default to 0.5 / 0.5
  (config.FUSION_*_WEIGHT) and can be tuned once paired data exists.

Verdict
  MALICIOUS   fused ≥ FUSION_MALICIOUS_THRESHOLD  (default 0.70)
  SUSPICIOUS  fused ≥ FUSION_SUSPICIOUS_THRESHOLD (default 0.40)
  BENIGN      otherwise

Agreement category (why the verdict was reached — shown to the analyst)
  static high  = static_conf   ≥ CONFIDENCE_GREEN (0.80)
  dynamic high = dynamic_score ≥ 0.50

  | static high | dynamic high | category        | meaning                                              |
  |-------------|--------------|-----------------|------------------------------------------------------|
  | yes         | yes          | CONFIRMED       | known family, behaviour confirms it                  |
  | yes         | no           | STATIC_ONLY     | dormant/evasive sample, or static false positive     |
  | no          | yes          | NOVEL_VARIANT   | behaves maliciously but matches no known family —    |
  |             |              |                 | the variant case MalTwin is designed to catch        |
  | no          | no           | NO_EVIDENCE     | neither view shows malicious evidence                |

  NOVEL_VARIANT always escalates the verdict to at least SUSPICIOUS, even if
  the weighted average falls below the threshold, because missing an unseen
  variant is the costlier error in an IIoT setting.

Status for Evaluation 3
  The fusion logic is complete and unit-tested. Truly *paired* evaluation
  (the same binary's image AND its runtime traffic) needs the Digital Twin
  (Module 1) to execute samples and capture traffic; that is scheduled for
  Evaluation 4. Until then the dashboard pairs a static result with a
  separately supplied capture and labels the pairing as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import config

CATEGORIES = ('CONFIRMED', 'STATIC_ONLY', 'NOVEL_VARIANT', 'NO_EVIDENCE',
              'STATIC_SOURCE_ONLY', 'DYNAMIC_SOURCE_ONLY')

CATEGORY_TEXT = {
    'CONFIRMED':     'Binary matches a known malware family and its runtime behaviour is malicious.',
    'STATIC_ONLY':   'Binary matches a known family but showed no malicious behaviour — '
                     'possibly dormant, sandbox-aware, or a static false positive.',
    'NOVEL_VARIANT': 'Runtime behaviour is malicious but the binary matches no known family '
                     'with confidence — possible new or obfuscated variant.',
    'NO_EVIDENCE':   'Neither static nor dynamic analysis found malicious evidence.',
    'STATIC_SOURCE_ONLY':  'Only static analysis available.',
    'DYNAMIC_SOURCE_ONLY': 'Only dynamic analysis available.',
}


@dataclass
class FusionResult:
    verdict: str
    fused_score: float
    category: str
    explanation: str
    static_score: Optional[float]
    dynamic_score: Optional[float]
    weights: dict
    family: Optional[str] = None
    dominant_behaviour: Optional[str] = None
    mitre_techniques: list = field(default_factory=list)
    escalated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _verdict(score: float, mal_t: float, sus_t: float) -> str:
    if score >= mal_t:
        return 'MALICIOUS'
    if score >= sus_t:
        return 'SUSPICIOUS'
    return 'BENIGN'


def merge_mitre_sources(static_mitre: Optional[dict], dynamic_mitre: Optional[dict]) -> list[dict]:
    """Union of techniques, each tagged with the evidence source(s) that support it."""
    merged: dict[str, dict] = {}
    for src, m in (('static', static_mitre), ('dynamic', dynamic_mitre)):
        for t in (m or {}).get('techniques', []):
            entry = merged.setdefault(t['id'], {'id': t['id'], 'name': t['name'], 'sources': []})
            if src not in entry['sources']:
                entry['sources'].append(src)
    # Techniques corroborated by both sources first
    return sorted(merged.values(), key=lambda e: (-len(e['sources']), e['id']))


def fuse(
    static_conf: Optional[float] = None,
    dynamic_score: Optional[float] = None,
    family: Optional[str] = None,
    dominant_behaviour: Optional[str] = None,
    static_mitre: Optional[dict] = None,
    dynamic_mitre: Optional[dict] = None,
    w_static: float = config.FUSION_STATIC_WEIGHT,
    w_dynamic: float = config.FUSION_DYNAMIC_WEIGHT,
    static_high: float = config.CONFIDENCE_GREEN,
    dynamic_high: float = 0.5,
    malicious_t: float = config.FUSION_MALICIOUS_THRESHOLD,
    suspicious_t: float = config.FUSION_SUSPICIOUS_THRESHOLD,
) -> FusionResult:
    """Combine static and dynamic evidence into one verdict. See module docstring."""
    if static_conf is None and dynamic_score is None:
        raise ValueError("At least one of static_conf or dynamic_score is required")
    for name, v in (('static_conf', static_conf), ('dynamic_score', dynamic_score)):
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if w_static < 0 or w_dynamic < 0:
        raise ValueError("Fusion weights must be non-negative")

    parts, wsum = 0.0, 0.0
    if static_conf is not None:
        parts += w_static * static_conf; wsum += w_static
    if dynamic_score is not None:
        parts += w_dynamic * dynamic_score; wsum += w_dynamic
    if wsum == 0:
        raise ValueError("Weights of the available sources sum to zero")
    fused = parts / wsum

    if static_conf is not None and dynamic_score is not None:
        s_hi, d_hi = static_conf >= static_high, dynamic_score >= dynamic_high
        category = ('CONFIRMED' if s_hi and d_hi else
                    'STATIC_ONLY' if s_hi else
                    'NOVEL_VARIANT' if d_hi else 'NO_EVIDENCE')
    elif static_conf is not None:
        category = 'STATIC_SOURCE_ONLY'
    else:
        category = 'DYNAMIC_SOURCE_ONLY'

    verdict = _verdict(fused, malicious_t, suspicious_t)
    escalated = False
    if category == 'NOVEL_VARIANT' and verdict == 'BENIGN':
        verdict, escalated = 'SUSPICIOUS', True

    return FusionResult(
        verdict=verdict,
        fused_score=float(fused),
        category=category,
        explanation=CATEGORY_TEXT[category] + (' Verdict escalated to SUSPICIOUS.' if escalated else ''),
        static_score=static_conf,
        dynamic_score=dynamic_score,
        weights={'static': w_static, 'dynamic': w_dynamic},
        family=family,
        dominant_behaviour=dominant_behaviour,
        mitre_techniques=merge_mitre_sources(static_mitre, dynamic_mitre),
        escalated=escalated,
    )
