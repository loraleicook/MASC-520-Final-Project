# Post-hoc error diagnostics

These calculations describe existing predictions. They do not fit new models, change hyperparameters, remove test images or replace the primary eight-image evaluation.

## 1. Error concentration

The two TPU views have hardness 35.26 Shore D. They are two images of one material condition, not two verified independent specimens. Both lie below the training target range of 67.32–80.76.

| Model | TPU share of total squared error | TPU MAE | Other six images MAE |
|---|---:|---:|---:|
| Training mean | 95.35% | 40.835 | 4.621 |
| Linear regression | 87.24% | 37.712 | 7.531 |
| Ridge fixed | 89.62% | 39.004 | 6.232 |
| RF baseline | 94.91% | 41.405 | 4.796 |
| Ridge tuned | 90.97% | 40.031 | 5.835 |
| RF tuned | 94.69% | 41.506 | 4.861 |
| CNN historical | 93.27% | 39.488 | 5.066 |

MAE uses Shore D units. Historical CNN TPU error accounts for about 93.27% of its test squared error. This locates the observed error concentration but does not isolate its unique cause. The other-six-image MAE is a descriptive breakdown, not a new test score or a substitute for reporting all eight images.

## 2. A mathematical ceiling for the current random forests

The current direct-target random forests use ordinary leaf means and averages across trees. Their predictions therefore stay inside the training target range. The nearest allowed prediction for each TPU image is at least 67.32, a difference of 32.06 from its target.

Even if predictions for all other test images were exact:

- The smallest possible squared-error sum is 2 × (67.32 − 35.26)² = 2055.6872.
- The test total sum of squares around its own mean, 62.42, is 2001.7472.
- The optimistic R² ceiling is 1 − 2055.6872 / 2001.7472 = −0.02694646.

This ceiling is not an achieved score. It applies to this standard RF setup with these targets and fixed split. It does not apply to Ridge, ordinary linear regression or CNN, and it does not establish that SEM cannot predict hardness under other conditions.

## 3. Unseen magnification categories

The categorical magnification encoder fits only the training categories, 60 and 500. Both test categories, 250 and 1000, are unseen. With `handle_unknown='ignore'`, all eight test rows receive the same all-zero code for the magnification field.

This makes the magnification contribution identical across the two unseen categories. Image-derived features remain different where their values differ. The separate effect of this encoding choice has not been isolated. The upload package documents the existing experiment and does not change the encoder after observing test scores.

## 4. R² and the mean baseline

Test R² uses the mean of the evaluated true test targets as its mathematical reference. Here that test mean is 62.42 Shore D. The practical training-mean benchmark predicts 76.0947368421, the training mean, for every test image. These are different references.

A negative test R² means squared error exceeds the test-mean reference error. It does not by itself mean performance is worse than the separately evaluated training-mean predictor.

## 5. What the checks establish

`scripts/check_results.py` recomputes all seven test metric rows and checks all 56 saved per-image error records. It verifies cached feature/prediction filename alignment and target/magnification values, the recorded selection by training CV MAE, the RF ceiling, and the actual unseen-category encoding.

These are arithmetic and cached-data consistency checks. They do not independently verify raw-image pairing, hardness measurement units against every original source, physical specimen identities, or the supplied raw-image hashes. Condition groups remain proxies. The test set was previously evaluated. No statistical-significance claim follows from these checks.

Run from the module directory:

```bash
python scripts/check_results.py --check
```

To regenerate the two diagnostic outputs, without fitting any model:

```bash
python scripts/check_results.py --write
```

The JSON and CSV outputs are in `results/posthoc_diagnostics.*`.

## References

- [scikit-learn: R² score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- [scikit-learn: RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
