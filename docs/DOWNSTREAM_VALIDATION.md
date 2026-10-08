# Validation of the GitHub-layout submission

Platform: Windows, Python 3.12.10, CPU. Dependency versions: `requirements-tested.txt`.

Completed on 2026-10-08: source-integrity verification passed, all 17 submitted
entry-point `--help` checks exited successfully, and the synthetic smoke run passed.

Source integrity: 65 original files are mapped into the native upstream layout.
23 files require only recorded import or path-string changes; all other files are
byte-identical to their working copies. The exact replacements are checked against
`D:/last` by `Others/verify_downstream_layout.py`.

Execution checks use the relocated original implementations with synthetic inputs,
48 clinical samples, two folds and small random checkpoints. Clinical binary/multiclass
classification and regression, three baseline methods, and the event trainer are
exercised successfully. The event trainer ran two epochs and exported predictions. No research
algorithm or hyperparameter defaults are modified for this check; reduced settings
are supplied in separate temporary smoke configs.

A smoke run validates execution only. GPU/multi-GPU, full cohort performance, original
trained weights, optional expanded IGLU and manuscript metric reproduction are not
covered. Local real datasets and experiment outputs remain untouched.
