# Planned version-1 data contracts

All paths are POSIX-style relative to a configured data/artifact root; reject absolute paths, traversal, missing files, duplicate IDs and invalid labels. SHA256 is lowercase 64-character hex. JSON/YAML are UTF-8; CSV includes a header. Schema version changes require a migration note and affected-owner review. Artifacts are not committed by default.

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
- Split metadata JSON records algorithm/version, seed 42, ratios, actual counts, source archive hash, group audit and manifest hash.

Proposed API: `validate_dataset(root, mapping) -> validation_report`; `create_splits(validated_manifest, split_config) -> split_manifest`. These signatures are design contracts only.

## Preprocessing contract

Raw files remain unchanged. RGB replication after grayscale conversion; full-image bicubic resize, no random crop. Diffusion tensors: float32/selected training dtype, CHW 3×512×512, normalized [-1,1]. Classifier: 3×224×224, ImageNet weight-enum normalization; training-only horizontal/vertical flip probabilities 0.5, no color jitter. Validate these augmentations against defect semantics before freeze. Evaluation is deterministic. Cache key includes input hash and transform-config hash.

## Adapter and synthetic manifests

Adapter metadata JSON: `schema_version,run_id,git_commit,config_sha256,train_manifest_sha256,base_model_id,base_revision,base_weights_sha256,adapter_sha256,trainable_modules,rank,alpha,steps,seed,environment_path,license_record`.

Synthetic JSONL: `schema_version,image_id,relative_path,sha256,class_id,class_name,prompt,negative_prompt,seed,base_revision,adapter_sha256,scheduler,inference_steps,guidance_scale,width,height,run_id,parent_ids,mask_id,quality_status,rejection_reason,reviewer_id`.

`parent_ids=[]` and `mask_id=null` for text-to-image; optional inpainting parents/masks must resolve exclusively to training-approved sources. `quality_status` is `pending`, `accepted` or `rejected`. All candidates remain in the ledger; training loader admits only accepted records. Class prompts are intended labels until review. Synthetic records can never have real-test membership.

## Training and evaluation artifacts

Run record follows [template](../../configs/experiment-record.yaml). Predictions CSV: `run_id,image_id,true_class,predicted_class,p0,p1,p2,p3,p4,p5`. Require exactly one row for every test ID, six finite probabilities in [0,1] summing to 1 within tolerance, consistent argmax and class mapping. Metrics JSON stores schema, counts, class order, macro metrics, per-class support, confusion counts and all input hashes. Use JSON `null` for unavailable results, never fabricated zeroes.

Directories: `artifacts/<stage>/<run_id>/`, run ID `YYYYMMDDTHHMMSSZ_<stage>_<seed>_<shortcommit>`. No overwriting completed runs. Record interrupted runs explicitly and hash resumed checkpoints. Local `metrics.json`, `predictions.csv`, `resolved-config.yaml`, `environment.txt`, and `run.json` support independent auditing.
