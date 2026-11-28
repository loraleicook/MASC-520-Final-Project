## Additional baseline optimization and evaluation

The [Person 3 evaluation extension](person3_evaluation/README.md) adds completed baseline training, tuning, grouped validation and reproducible figures while preserving the original team analysis.

Start with the [executed notebook](person3_evaluation/MASC_520_Person3_Completed.ipynb). Its feature data and actual outputs are embedded, so viewing results requires no rerun or image upload. Reproduction starts from cached features rather than raw SEM images. For local use, follow the folder README and launch inside person3_evaluation to avoid the repository root's package-name collisions.

Six baseline strategies used 76 training images, with nested five-outer/three-inner validation across 28 material-treatment condition groups. RF baseline had the lowest CV MAE, 2.467. Ridge tuning reduced CV MAE by 6.9%, but held-out improvements were inconsistent. Historical CNN and training-mean test MAE were nearly equal, 13.672 and 13.675.

Groups are condition proxies, not verified specimens. The previously viewed eight-image test set differs in source, composition, magnification and hardness coverage. CNN predictions remain historical and were not retrained.

See [results](person3_evaluation/results/), [figures](person3_evaluation/figures/), [diagnostics](person3_evaluation/docs/DIAGNOSTICS.md) and [reproduction instructions](person3_evaluation/README.md). This extension does not replace the original notebook or establish reliable cross-study generalization.
