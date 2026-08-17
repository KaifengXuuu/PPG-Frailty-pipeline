"""Effect-size ShapeFormer 实验门面 / Canonical experimental ShapeFormer facade.

English: Expose only the explicitly selected, outer-fold-bound effect-size discovery
route and its patch-first masked-attention model; this facade is not PISD parity.

中文：仅暴露显式选择、绑定 outer fold 的效应量发现路线及其 patch-first
掩码注意力模型；本门面不构成 PISD parity。
"""

from .shapeformer import EffectSizeShapelets, ExperimentalShapeFormer, discover_effect_size_shapelets

__all__ = ["EffectSizeShapelets", "ExperimentalShapeFormer", "discover_effect_size_shapelets"]
