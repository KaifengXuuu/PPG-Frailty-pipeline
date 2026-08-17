"""质量与 signal route 约束 / Quality-to-signal-route constraints.

中文：验证 rate-only 的形态学不适用状态。English: Validate not-applicable morphology on rate-only routes.
"""

from ..contracts import QualityResult, SignalRoute


def assert_quality_route(result: QualityResult, route: SignalRoute | str) -> None:
    """强制 rate-only 的 Q_morph 不适用 / Enforce not-applicable morphology."""

    resolved = route if isinstance(route, SignalRoute) else SignalRoute(route)
    result.validate_for_route(resolved)


__all__ = ["assert_quality_route"]
