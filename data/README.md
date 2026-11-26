# Dataset and storage plan

## NEU-CLS provenance record

| Field | Evidence or status |
| --- | --- |
| Publisher | Northeastern University faculty page for Kechen Song and Yunhui Yan |
| Source and access date | [Official NEU database page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/), reviewed 2026-11-13 |
| Distribution | [NEU-CLS.rar on Google Drive](https://drive.google.com/file/d/1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc/view), linked by the official page alongside a Baidu option |
| Expected contents | Image-only classification collection of hot-rolled steel defects. The publisher describes 1,800 grayscale images at 200 × 200 pixels, 300 per class. |
| Archive inspection | Pending. No local archive was found. The Drive preview identified the archive, but its bytes and internal folder layout were not inspected. |
| Acquisition date, release ID and SHA-256 | Pending. A page access date is distinct from acquisition. No release version or checksum was verified. |
| Use and sharing | Blocked pending written terms covering the intended use. The official page provides downloads and requests citation, but does not state an explicit license or permission for redistribution. Archive terms remain unverified. |

| Publisher class | Publisher abbreviation | Expected images |
| --- | --- | ---: |
| Crazing | Cr | 300 |
| Inclusion | In | 300 |
| Patches | Pa | 300 |
| Pitted surface | PS | 300 |
| Rolled-in scale | RS | 300 |
| Scratches | Sc | 300 |

These abbreviations describe the publisher's classes. Confirm actual filenames and folders from the approved archive before relying on any naming convention. The source lists NEU-CLS-64 separately. NEU-DET includes bounding boxes for detection, and NEU-SEG is a separate segmentation collection. Third-party mirrors and their split folders are not evidence of the official NEU-CLS layout.

Before using the data, obtain the distributor's terms or written confirmation covering internal acquisition and model training. Ask separately about sharing raw images, example figures and derived artifacts. Keep those permissions with the provenance record. The [project license ledger](../docs/research/licensing.md) separates the dataset from the code license.

Once an approved copy is available, record the original filename, source URL, acquisition date and size. On Windows, run `Get-FileHash -Algorithm SHA256 -LiteralPath 'path-to-NEU-CLS.rar'` twice on the original archive and compare the digests. Record the digest and archive listing. Inspect entries for absolute paths and parent-directory traversal before extraction. Keep the archive and extracted images outside Git.

Publisher-requested citation relevant to classification: K. Song and Y. Yan (2013), “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” *Applied Surface Science*, 285, 858–864. See the official page for its complete citation list.

## Local storage

### Freeze reviewed splits (#4)

Real freezing is pending. Keep the source read-only and confirm dataset permission before use.
Run the audit below first. Then obtain its canonical digest:

```powershell
python scripts/freeze_splits.py --audit artifacts/neu-audit.json --audit-digest
```

Save a peer-reviewed JSON file using the schema below. Replace the illustrative values
with decisions for the actual audit. Record every perceptual candidate as `same` or
`different`, including pairs involving excluded files because they can connect other images.
Exact byte matches are always grouped. Use `source_groups` for additional confirmed
capture siblings. Record explicit exclusions with reasons and explain any count, structure,
or inventory findings in `inventory_note`. Fix source layout or labels through a fresh audit
if needed. Files with proposed exclusions cannot be silently admitted.

```json
{
  "schema_version": 1,
  "audit_sha256": "<audit digest>",
  "reviewer": "<peer reviewer>",
  "inventory_note": "<resolution of inventory findings, if any>",
  "exclusions": {},
  "near_duplicates": [],
  "source_groups": []
}
```

Each `near_duplicates` entry has `paths` (the two paths from the audit) and `decision`.
Each `source_groups` entry is an array of relative paths.

```powershell
python scripts/freeze_splits.py --root "path/to/extracted/NEU-CLS" --audit artifacts/neu-audit.json --review artifacts/neu-review.json --allow-unresolved-dhash --output artifacts/neu-split-v1
```

The freezer computes `dataset_source_sha256` from sorted relative image paths and
actual file SHA-256 values and verifies them against the audit. The audit digest is
stored separately; do not pass an archive hash to the freezer. The opt-in flag is
appropriate only when the review record contains the documented amendment and
explicitly preserves unresolved dHash candidates as unverified and ungrouped.

Use the same `--inventory` if one was supplied to the audit. The CLI re-runs #3 against
the source and refuses a changed report. An identical rerun verifies the existing output
byte for byte. A changed or incomplete output requires a new directory.

Downstream readers use `manifest.csv` and `split-metadata.json` together. The metadata
contains seed, ratios, audit and manifest hashes, counts and review decisions. Allocation
works within each class, places larger groups first and uses the seed to break group-order
ties. Indivisible groups can move the result away from 70/15/15. Each class needs at least
three independent groups. Peer approval of the final test split is still required before
training. Synthetic fixtures exercise the pipeline and provide no evidence about NEU itself.

### Run the CPU audit (#3)

Real NEU-CLS verification is pending. Use an approved, extracted copy after resolving the terms above. Point the audit at the image root. The supported layouts are flat files such as `Cr_1.bmp` or a single level of canonical/alias class folders. Names are case-insensitive. A recognized filename prefix that disagrees with its class folder is flagged. Wrapper directories and pre-made split folders are reported for review.

```powershell
python -m pip install -e ".[dev,audit]"
New-Item -ItemType Directory -Force artifacts
python scripts/audit_dataset.py --root "path/to/extracted/NEU-CLS" --report artifacts/neu-audit.json
python -m pytest -q
```

Use a fresh report filename outside the source directory. The audit writes JSON and prints a summary. Exit 0 means the checks found no issues, 1 means review is needed, and 2 means the audit could not complete. Supply `--inventory expected-paths.json` with a trusted JSON array of root-relative paths to identify specific missing files. Without it, class count shortages are visible but missing filenames are unknown.

The audit reads single-frame BMP, PNG, JPEG and TIFF files and checks original mode `L` and size 200 × 200. It skips linked entries and refuses inputs above 8 MiB or 4 million pixels. The report distinguishes valid files, files needing review, and proposed exclusions. It never deletes or relabels source files. SHA-256 identifies identical bytes. The 64-bit difference hash compares coarse brightness patterns with a default distance of 5 bits. Similar steel textures can collide, so review each suggested match and resolve conflicts before grouping or splitting.

Planned ignored layout: `data/local/raw/<source-fingerprint>/`, `data/local/manifests/<split-version>/`, `data/local/cache/<transform-hash>/`, `data/local/synthetic/<run-id>/`. Preserve raw data read-only. Store source URL/date, an archive hash only when the original archive is available, the derived source fingerprint, per-file hashes, labels, grouping/exclusions and split metadata. Back up small manifests with the corresponding dataset in approved team storage.

Canonical schemas are in [contracts](../docs/architecture/contracts.md). Do not commit data, masks, generated images or checkpoints. NEU-SEG requires a separate overlap/ancestry/terms review before optional use.
