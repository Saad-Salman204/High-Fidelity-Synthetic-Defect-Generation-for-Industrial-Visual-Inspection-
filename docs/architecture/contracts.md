# Version-1 data contracts

All paths are POSIX-style relative to a configured data/artifact root; reject absolute paths, traversal, missing files, duplicate IDs and invalid labels. SHA256 is lowercase 64-character hex. JSON/YAML are UTF-8; CSV includes a header. Schema version changes require a migration note and affected-owner review. Research artifacts remain local by default; only the explicitly verified aggregate research bundle and small exploratory gallery are prepared for publication.

The lab launcher stores version-1 local session JSON containing paths and hash-bound human review receipts in ignored artifact storage. Scientific parameters remain in the existing validated experiment YAML/JSON. Config templates are proposed copies. Launcher logs record command/exit/approval events, and backups publish a completion index only after byte verification. The launcher reuses stage CLIs and their checkpoint, review and split contracts.

## Class mapping

| ID | Canonical label | Source alias examples (verify archive) |
| --- | --- | --- |
| 0 | crazing | Cr |
| 1 | inclusion | In |
| 2 | patches | Pa |
| 3 | pitted_surface | PS |
| 4 | rolled_in_scale | RS |
| 5 | scratches | Sc |

Never infer class order from directory enumeration. Persist mapping and its hash with every run.

## Real manifest CSV

Required fields: `schema_version,image_id,relative_path,sha256,class_id,class_name,width,height,mode,group_id,split,source_version`.

- `schema_version=1`; immutable `image_id` derives from source identity; `sha256` hashes original bytes.
- `split` is exactly `train`, `val` or `test`; every raw image occurs once or appears in a separate exclusions ledger.
- `group_id` joins byte duplicates, reviewed near-duplicates and known capture/source siblings. No group crosses splits.
- Mode/size record original decode properties, expected L/200/200; deviations require review, not silent resizing during audit.
- Split metadata JSON records algorithm/version, seed 42, ratios, actual counts, deterministic `dataset_source_sha256`, separate audit/review digests, unresolved dHash candidates, group audit and manifest hash. The source fingerprint is computed from sorted relative image paths and verified image-file SHA-256 values; it is not an archive hash.

The #3 API is `validate_dataset(root, *, expected_paths=None, near_threshold=5) -> dict`. It applies the fixed class mapping above. Its JSON report records schema version, per-file source paths, class IDs, original SHA-256 and decode metadata where available, findings, dispositions, observed counts and inventory shortages. Counts describe discovered image candidates before exclusions. Exact groups and perceptual pairs contain member paths and pending review status. `freeze_splits` requires a review record; strict mode rejects unresolved dHash pairs. The documented opt-in amendment keeps those pairs explicitly unverified and ungrouped in split metadata. The freezer computes `dataset_source_sha256` from sorted relative paths and verified source-image hashes, checks the audit hashes, and stores `audit_sha256` separately.

## Preprocessing contract

Raw files remain unchanged. RGB replication after grayscale conversion; full-image bicubic resize, no random crop. Diffusion tensors: float32/selected training dtype, CHW 3×512×512, normalized [-1,1]. Classifier: 3×224×224, ImageNet weight-enum normalization; training-only horizontal/vertical flip probabilities 0.5, no color jitter. Validate these augmentations against defect semantics before freeze. Evaluation is deterministic. The loader decodes from source on each access. A future cache must key by source and transform hashes.

Use `read_frozen_manifest` to verify the CSV hash and group isolation. `NEUManifestDataset` selects one partition and checks each source hash on access. Samples carry `image`, integer `label` and stable `image_id`. `make_loader` preserves evaluation order. The shared `preprocess_image` transform also accepts approved synthetic images. Real sample-grid review is pending.

## Adapter and synthetic manifests


Adapter metadata JSON: `schema_version,run_id,git_commit,config_sha256,train_manifest_sha256,base_model_id,base_revision,base_weights_sha256,adapter_sha256,trainable_modules,rank,alpha,steps,seed,environment_path,license_record`.

Synthetic JSONL: `schema_version,image_id,relative_path,sha256,class_id,class_name,prompt,negative_prompt,seed,base_revision,adapter_sha256,scheduler,inference_steps,guidance_scale,width,height,run_id,parent_ids,mask_id,quality_status,rejection_reason,reviewer_id`.

`parent_ids=[]` and `mask_id=null` for text-to-image; optional inpainting parents/masks must resolve exclusively to training-approved sources. `quality_status` is `pending`, `accepted` or `rejected`. All candidates remain in the ledger. The human-review training loader admits only accepted records; the separate opt-in automated-selection loader admits only candidates bearing the documented policy marker while preserving pending status. Class prompts are intended labels until review. Synthetic records can never have real-test membership.

The #13 quality evaluator consumes the #12 review export and verified frozen split. It reads only real training pixels and scores only accepted synthetic rows. Version 1 `quality.json` records source and review hashes, extractor identity, preprocessing, sampling parameters, overall and per-class KID status and statistics, grid image IDs, and reviewer observations. Unavailable statistics are null with explicit reasons. The evaluator never writes to the review ledger.

The [generation CLI](../experiments/generate-candidates.md) writes pending rows
with `real_manifest_sha256` and adapter checkpoint provenance. #12 reviews those
rows and preserves their provenance in a separate accepted-only export for #14.
The current reader supports text-to-image with empty ancestry and a null mask.
See the [classifier arm guide](../experiments/classifier-arms.md) for the handoff.

## Training and evaluation artifacts

Run record follows [template](../../configs/experiment-record.yaml). Predictions CSV: `run_id,image_id,true_class,predicted_class,p0,p1,p2,p3,p4,p5`. Require exactly one row for every test ID, six finite probabilities in [0,1] summing to 1 within tolerance, consistent argmax and class mapping. Metrics JSON stores schema, counts, class order, macro metrics, per-class support, confusion counts and all input hashes. Use JSON `null` for unavailable results, never fabricated zeroes.

Directories: `artifacts/<stage>/<run_id>/`, run ID `YYYYMMDDTHHMMSSZ_<stage>_<seed>_<shortcommit>`. No overwriting completed runs. Record interrupted runs explicitly and hash resumed checkpoints. Local `metrics.json`, `predictions.csv`, `resolved-config.yaml`, `environment.txt`, and `run.json` support independent auditing.
