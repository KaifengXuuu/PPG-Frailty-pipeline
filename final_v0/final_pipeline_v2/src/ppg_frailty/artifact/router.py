"""规范 singular artifact 路由门面 / Canonical singular artifact router.

中文：实现暂存于历史 plural 包，但正式 pipeline、CLI 与注册表只能从本门面
导入。这样 reducer factory 与无 fallback route 只有一个可见规范入口。
English: Implementations remain in the historical plural package, while formal
pipeline/CLI code imports this sole singular facade.
"""

from ..artifacts.router import (
    ArtifactRouteOutcome,
    UnsupportedReducer,
    get_reducer,
    run_artifact_route,
)

__all__ = [
    "ArtifactRouteOutcome",
    "UnsupportedReducer",
    "get_reducer",
    "run_artifact_route",
]
