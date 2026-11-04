# Security and artifact handling

The project is a research scaffold, not a deployed service. Keep credentials in local environment/credential stores, never `.env.example` or notebook outputs. Dataset, pretrained and generated artifacts are ignored by default; use approved local/team storage with access controls and backups.

Verify download provenance and hashes. Inspect archive members before extraction; reject path traversal. Prefer safetensors when supported; loading untrusted serialized Python objects can execute code. Do not run downloaded training scripts without review. Review dependency sources and model/dataset terms separately.

Report suspected credentials, unsafe loading or leakage privately to the repository owner through the team's established channel or GitHub private vulnerability reporting if enabled. Do not attach secrets/data to public issues. If a credential is exposed, revoke/rotate it first, notify the owner and coordinate any history repair; deleting a file is not sufficient. No security response SLA is promised by this student project.
