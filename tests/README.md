# Planned test strategy

No ML tests or implementations exist yet; no test result is claimed. I19 enables pytest once meaningful tests exist. Each implementation issue adds its own relevant tests rather than deferring correctness to I19.

Use small authored image/manifest fixtures for corrupt decode, grouping, split overlap, normalization, sampler membership, config validation and hand-computed metrics. Verify failure paths. No downloaded NEU images or pretrained weights in ordinary CI. Separate mocked sampler tests, CPU integration and real GPU pilot evidence. Never add dummy passing tests merely to obtain a green badge.
