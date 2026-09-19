"""Conservative composition of child certificates."""

from __future__ import annotations

from typing import Sequence

from .compiler import DEFAULT_DIRECTIONS, classify_interval, failure_safe_certificate
from .models import MobilityCertificate, MobilityInterval


def finite_hypothesis_hull(
    certificates: Sequence[MobilityCertificate], parent_ids: Sequence[str] = ()
) -> MobilityCertificate:
    children = tuple(certificates)
    if not children:
        return failure_safe_certificate("empty finite hypothesis set")
    names = set(children[0].directions)
    if any(set(child.directions) != names for child in children):
        return failure_safe_certificate("child certificates have incompatible directions")
    intervals = {}
    for name in sorted(names):
        lower = min(child.directions[name].lower for child in children)
        upper = max(child.directions[name].upper for child in children)
        intervals[name] = MobilityInterval(lower, upper, classify_interval(lower, upper))
    status = "failure_safe" if any(child.status == "failure_safe" for child in children) else "finite_hull"
    reasons = tuple(
        reason for child in children for reason in child.reasons
    )
    return MobilityCertificate(
        directions=intervals,
        status=status,
        reasons=reasons,
        parent_ids=tuple(parent_ids),
        parameters={"child_count": len(children)},
    )


def unresolved_continuous_uncertainty(reason: str) -> MobilityCertificate:
    """M0 cannot certify continuous boxes; fail closed until M1 exists."""

    return failure_safe_certificate(f"unresolved_continuous_uncertainty: {reason}", DEFAULT_DIRECTIONS)
