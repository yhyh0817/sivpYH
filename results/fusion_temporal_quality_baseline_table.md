# Fusion Temporal Quality Baseline Table

This table is generated from saved validation JSON files.  The ResNet-18 feaCD row is a local probe, not an official TemCoCo reproduction.

### VidLLVIP independent

| Group | Method | AUROC | Worst AUROC | Accuracy | FPR | Spearman |
|---|---|---:|---:|---:|---:|---:|
| ours | Ours severity head | 0.9948 | 0.9901 | 0.9141 | 0.0417 | 0.9262 |
| ours | Ours composite | 0.9988 | 0.9981 | 0.8510 | 0.0639 | 0.9004 |
| ours | Ours detection head | 0.9977 | 0.9966 | 0.9760 | 0.0056 | 0.8406 |
| ablation | Span residual | 0.9981 | 0.9971 | 0.9735 | 0.0417 | 0.8631 |
| ablation | Static operator residual | 0.9494 | 0.9242 | 0.7828 | 0.1306 | 0.8433 |
| ablation | Source-conditioned residual | 0.9977 | 0.9966 | 0.9722 | 0.0444 | 0.8406 |
| baseline | Global weight drift | 0.8180 | 0.6490 | 0.6919 | 0.0583 | 0.7753 |
| baseline | SEA-RAFT flowD visible-source probe | 0.3084 | 0.2680 | 0.4495 | 0.0111 | 0.6787 |
| baseline | Temporal difference | 0.3108 | 0.2708 | 0.4482 | 0.0139 | 0.6541 |
| baseline | Flow warp error | 0.3108 | 0.2708 | 0.4482 | 0.0139 | 0.6541 |
| baseline | TemCoCo-style flowD visible | 0.3108 | 0.2708 | 0.4482 | 0.0139 | 0.6541 |
| baseline | Entropy instability | 0.3816 | 0.3468 | 0.4129 | 0.0917 | 0.6084 |
| baseline | Standard-deviation instability | 0.3840 | 0.3466 | 0.3977 | 0.1278 | 0.5942 |
| baseline | Source-normalized change | 0.8118 | 0.7713 | 0.7210 | 0.3250 | 0.5782 |
| baseline | Spatial-frequency instability | 0.3879 | 0.3497 | 0.4104 | 0.0972 | 0.5738 |
| baseline | Mutual-information instability | 0.5199 | 0.3732 | 0.4785 | 0.0306 | 0.4850 |
| baseline | TemCoCo-style feaCD Sobel | 0.6619 | 0.6431 | 0.4646 | 0.0778 | 0.3590 |
| baseline | Edge-correlation instability | 0.4226 | 0.3950 | 0.4508 | 0.0083 | 0.2734 |
| baseline | ResNet-18 feaCD visible-source probe | 0.6681 | 0.6660 | 0.4874 | 0.1556 | 0.1146 |

### M3SVD 12-pair

| Group | Method | AUROC | Worst AUROC | Accuracy | FPR | Spearman |
|---|---|---:|---:|---:|---:|---:|
| ours | Ours severity head | 0.9926 | 0.9861 | 0.9457 | 0.0833 | 0.8666 |
| ours | Ours detection head | 0.9991 | 0.9986 | 0.9874 | 0.0000 | 0.8055 |
| ours | Ours composite | 0.9827 | 0.9691 | 0.8965 | 0.1694 | 0.7873 |
| ablation | Span residual | 0.9992 | 0.9984 | 0.9823 | 0.0306 | 0.8352 |
| ablation | Source-conditioned residual | 0.9991 | 0.9986 | 0.9773 | 0.0417 | 0.8055 |
| ablation | Static operator residual | 0.9310 | 0.9090 | 0.8043 | 0.1806 | 0.7248 |
| baseline | Global weight drift | 0.8179 | 0.6381 | 0.7197 | 0.0944 | 0.7662 |
| baseline | Temporal difference | 0.2909 | 0.2592 | 0.4053 | 0.1083 | 0.6901 |
| baseline | Flow warp error | 0.2909 | 0.2592 | 0.4053 | 0.1083 | 0.6901 |
| baseline | TemCoCo-style flowD visible | 0.2909 | 0.2592 | 0.4053 | 0.1083 | 0.6901 |
| baseline | Source-normalized change | 0.8163 | 0.7961 | 0.7740 | 0.2056 | 0.5634 |
| baseline | TemCoCo-style feaCD Sobel | 0.7546 | 0.7526 | 0.5025 | 0.0667 | 0.5482 |

### M3SVD 30-pair

| Group | Method | AUROC | Worst AUROC | Accuracy | FPR | Spearman |
|---|---|---:|---:|---:|---:|---:|
| ours | Ours severity head | 0.9715 | 0.9456 | 0.9197 | 0.0825 | 0.8682 |
| ours | Ours composite | 0.9908 | 0.9867 | 0.9443 | 0.0308 | 0.8532 |
| ours | Ours detection head | 0.9963 | 0.9950 | 0.9527 | 0.0083 | 0.8181 |
| ablation | Source-conditioned residual | 0.9963 | 0.9950 | 0.9534 | 0.0475 | 0.8181 |
| ablation | Span residual | 0.9959 | 0.9949 | 0.9485 | 0.0575 | 0.8162 |
| ablation | Static operator residual | 0.9383 | 0.9182 | 0.8174 | 0.0283 | 0.7945 |
| baseline | Global weight drift | 0.8250 | 0.6507 | 0.7383 | 0.1125 | 0.7390 |
| baseline | SEA-RAFT flowD visible-source probe | 0.2946 | 0.2606 | 0.4409 | 0.0300 | 0.6755 |
| baseline | Temporal difference | 0.2963 | 0.2628 | 0.4390 | 0.0342 | 0.6541 |
| baseline | Flow warp error | 0.2963 | 0.2628 | 0.4390 | 0.0342 | 0.6541 |
| baseline | TemCoCo-style flowD visible | 0.2963 | 0.2628 | 0.4390 | 0.0342 | 0.6541 |
| baseline | Source-normalized change | 0.8086 | 0.7823 | 0.8386 | 0.0033 | 0.5675 |
| baseline | TemCoCo-style feaCD Sobel | 0.7021 | 0.6796 | 0.4742 | 0.0358 | 0.5061 |
