# Local validation report

Validated in the build environment on 2026-08-06.

## Source coverage

- Uploaded notebooks incorporated: 12
- Total code cells preserved: 73
- Newly incorporated notebooks: 4
- Newly incorporated code cells: 15

## Commands

```bash
python -m compileall -q src scripts examples tests extracted
PYTHONPATH=src python examples/quick_test.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Results

- Compilation return code: 0
- Quick-test return code: 0
- Unit-test return code: 0
- Credential scan findings excluding explicit REDACTED placeholders: 0

### Quick-test output

```text
{
  "native": {
    "overall_accuracy": 0.9375,
    "mean_iou": 0.75,
    "per_class_iou": [
      null,
      0.75,
      0.0,
      null,
      null,
      1.0,
      null,
      null,
      null,
      1.0,
      null,
      null,
      null,
      1.0,
      null,
      null
    ],
    "confusion": [
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        3,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    ]
  },
  "functional_after_coarsening": {
    "overall_accuracy": 1.0,
    "mean_iou": 1.0,
    "per_class_iou": [
      null,
      1.0,
      1.0,
      null,
      null,
      1.0,
      1.0
    ],
    "confusion": [
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        1,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        1,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        1,
        0
      ],
      [
        0,
        0,
        0,
        0,
        0,
        0,
        1
      ]
    ]
  },
  "decoded_classes": [
    1,
    5,
    9,
    13
  ]
}
PASS: wrote /mnt/data/GeoAI_GID_YRB_Real_Workflow_v2.1.0/outputs/quick_test_result.json
```

### Unit-test output

```text

Spreadsheet runtime warmup failed during python startup
Traceback (most recent call last):
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/patches/warm_spreadsheet_runtime_on_startup.py", line 26, in warm_spreadsheet_runtime_on_startup
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/spreadsheet_warmup.py", line 785, in warm_spreadsheet_runtime
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/spreadsheet_warmup.py", line 720, in _warm_feature_flows
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/spreadsheet_warmup.py", line 704, in _warm_collaboration_flows
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/generated/interface/models.py", line 30820, in hydrate_crdt_from_proto
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/rpc/remote.py", line 749, in __call__
  File "/tmp/tmp.yTcnQsZYiA/artifact_tool_v2-2.8.4/artifact_tool/rpc/client.py", line 150, in call
artifact_tool.rpc.client.RemoteError: hydrateCrdtFromProto requires an empty collaborative document.
test_metrics (test_core.CoreTests.test_metrics) ... ok
test_normalization (test_core.CoreTests.test_normalization) ... ok
test_palette_roundtrip (test_core.CoreTests.test_palette_roundtrip) ... ok
test_remap (test_core.CoreTests.test_remap) ... ok
test_resolution (test_core.CoreTests.test_resolution) ... ok
test_effects (test_factorial.FactorialTests.test_effects) ... ok
test_scene_summary (test_factorial.FactorialTests.test_scene_summary) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.004s

OK
```

## Boundary

The local verification covers package imports, deterministic label/resolution/metric
calculations, the complete factorial contrast implementation, and Python syntax for
all extracted and refactored scripts. Full GPU/data-dependent training and inference
were not executed because the required imagery, NPZ archives, checkpoints, pretrained
weights, and original compute environment were not supplied.
