"""MP-20 deterministic AE with unscaled RMSNorm (factor=1).

Controlled latent-normalization ablation.  This configuration is identical to
``mp_20_conifg_dict_large_h_rmsnorm.py`` except for ``rmsnorm_factor`` and the
isolated output tag.
"""

from copy import deepcopy

from config.FlowPrep.mp_20_conifg_dict_large_h_rmsnorm import conf as _scaled_conf


conf = deepcopy(_scaled_conf)
conf["model"]["rmsnorm_factor"] = 1.0
conf["result_version_tag"] = "rmsnorm"
