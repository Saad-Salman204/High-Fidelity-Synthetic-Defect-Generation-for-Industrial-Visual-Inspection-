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

Planned ignored layout: `data/local/raw/<archive-hash>/`, `data/local/manifests/<split-version>/`, `data/local/cache/<transform-hash>/`, `data/local/synthetic/<run-id>/`. Preserve raw data read-only. Store source URL/date/archive hash, per-file hashes, labels, grouping/exclusions and split metadata. Back up small manifests with the corresponding dataset in approved team storage.

Canonical schemas are in [contracts](../docs/architecture/contracts.md). Do not commit data, masks, generated images or checkpoints. NEU-SEG requires a separate overlap/ancestry/terms review before optional use.
