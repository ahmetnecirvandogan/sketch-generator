"""Sketch → PBR maps (albedo + roughness + lighting) ML scaffold.

Issue dependencies:
- #18 SH lighting projection — pbr_model/preprocess_lighting_sh.py
- #19 PyTorch data loader — pbr_model/dataset.py (this PR)
- #20 Model scaffold — pbr_model/model.py (next)
- #21 Training loop scaffold — pbr_model/train.py (after model)
"""
