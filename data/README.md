# Dataset and storage plan

No dataset is included or downloaded by repository initialization. Use official NEU-CLS after resolving terms, not NEU-DET bounding boxes as segmentation masks. See [provenance/license ledger](../docs/research/licensing.md).

Planned ignored layout: `data/local/raw/<archive-hash>/`, `data/local/manifests/<split-version>/`, `data/local/cache/<transform-hash>/`, `data/local/synthetic/<run-id>/`. Preserve raw data read-only. Store source URL/date/archive hash, per-file hashes, labels, grouping/exclusions and split metadata. Back up small manifests with the corresponding dataset in approved team storage.

Canonical schemas are in [contracts](../docs/architecture/contracts.md). Do not commit data, masks, generated images or checkpoints. NEU-SEG requires a separate overlap/ancestry/terms review before optional use.
