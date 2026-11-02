# Third-party runtime code

The experiment launchers use the following vendored or adapted implementations. Their upstream licenses are preserved beside the corresponding code.

- SyMat-derived implementation: https://github.com/divelab/AIRS/tree/main/OpenMat/SyMat (upstream GPL-3.0 text is preserved in `SyMat/LICENSE`; local adaptations differ from upstream).
- nflows: https://github.com/bayesiains/nflows (see `nflows/LICENSE.md` and the official sampler vendor copy).
- TabDDPM numerical diffusion implementation: https://github.com/yandex-research/tab-ddpm (see the vendor `LICENSE.md`).
- CDVAE: https://github.com/txie-93/cdvae
- DiffCSP: https://github.com/jiaor17/DiffCSP
- MatterGen and the adapted evaluation code: https://github.com/microsoft/mattergen
- FTCP: the original upstream files retain `third_party/FTCP/LICENSE`; `ftcp_recon` contains the local PyTorch adaptation used in the experiments.

Exact external-generator source revisions are recorded in `final_experiments/experiment_3/source_manifest.json`. DeLaC experiment-specific adapters are retained with the source on which the experiments depended.
