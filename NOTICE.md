# Attribution and third-party notices

This repository derives from [V2X-R](https://github.com/ylwhxht/V2X-R) and its OpenCOOD-based implementation. The upstream README, citations, source author comments and per-file license headers are retained. The local source commit and dataset revision are recorded in `repro/source_provenance.json`.

The source tree contains differing historical license declarations, including `TDG-Attribution-NonCommercial-NoDistrib` headers and a setup.py MIT metadata field. This packaging does not assign a new blanket license or resolve those differing declarations. Consult the original authors and applicable per-file terms before redistribution or commercial use. The repository is initially private.

V2X-R data is obtained directly from the authors' Hugging Face dataset at the pinned revision. Raw point clouds and annotations are not redistributed here. This project's trained selected checkpoints and its evaluation metadata are provided as GitHub Release assets. The original dataset README declares MIT; source-code terms remain separate.

Acknowledgements and citations for V2X-R, OpenCOOD, BM2CP, CoPerception and Where2comm remain in `docs/V2XR_UPSTREAM_README.md`. New DEER/SBR code, scientific protocol and experiment evidence are part of this project's source snapshot; no additional authorship or paper publication status is claimed by this packaging.

The supplemental four-channel L4DR adaptation in `opencood/project/l4dr.py` follows the authors' [L4DR code](https://github.com/ylwhxht/L4DR/tree/fcf2d7f0b7c632f9c23f75f720fe691be519d732), distributed under [Apache-2.0](https://github.com/ylwhxht/L4DR/blob/fcf2d7f0b7c632f9c23f75f720fe691be519d732/LICENSE). FAD/PointNet++ layer sizes, foreground supervision, bidirectional MME and MSGF are attributed to that implementation. The existing OpenPCDet PointNet++ CUDA operators retain their original source notices. Project data, detection-head and cooperative adaptations are described in the supplemental protocol.

The upstream README is stored at `docs/V2XR_UPSTREAM_README.md` to avoid a case-only collision with the project `README.md` on Windows.
