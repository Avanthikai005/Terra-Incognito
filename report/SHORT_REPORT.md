# Continual Learning for Cross-Regional Satellite-Based Disaster Response

**Format:** Short Technical Report (Max 4 Pages)  
**Authors:** Team Terra Incognita  
**Repository:** [Terra-Incognito](file:///c:/Users/NIVA%20KALYANI%20H/Downloads/Terra-Incognito)  

---

## Abstract
Automated building damage assessment from satellite imagery is critical for emergency disaster management. However, when vision models are fine-tuned sequentially on new disaster events (e.g., Hurricane → Tsunami → Wildfire), they suffer from **Catastrophic Forgetting**—overwriting visual representations of earlier geographical regions. Retraining from scratch on all historical imagery is computationally infeasible in time-critical rescue missions. In this report, we formulate disaster damage classification as a continual learning benchmark on Sentinel-2 satellite imagery. We evaluate four learning regimes and propose an **Experience Replay & Domain-Adaptive Buffer mechanism**. We further present an ablation showing that pre-computed backbone **domain distance** strongly predicts forgetting severity, allowing adaptive memory allocation.

---

## 1. Problem Statement & Motivation

During natural disasters, rescue agencies require rapid, automated building damage mapping from high-resolution satellite passes. While deep learning classifiers achieve high performance on static benchmarks, operational deployment faces three major hurdles:

1. **Catastrophic Forgetting in Sequential Deployment:** When a classifier trained on Caribbean hurricane damage (Task 1) is sequentially fine-tuned on Indonesian tsunami imagery (Task 2) and Californian wildfire data (Task 3), gradient updates overwrite earlier disaster features.
2. **Computational & Bandwidth Bottlenecks:** In emergency scenarios, satellite imagery arrives as localized data streams. Downloading massive historical datasets to retrain full joint models from scratch causes severe deployment delays.
3. **Domain Shift Across Disasters:** Different disaster types exhibit vastly different visual footprints (e.g., flood inundation vs. structural collapse vs. scorched earth), amplifying forgetting severity.

### Core Research Questions:
- **Q1:** How severe is catastrophic forgetting under standard naive sequential fine-tuning?
- **Q2:** How effectively can compact experience replay buffers protect past disaster memory?
- **Q3:** Can cheap, pre-computed feature domain distances predict forgetting severity and dynamically guide buffer sizing?

---

## 2. Methodology

### 2.1 Task Formulation & Data Preprocessing
We utilize the xBD satellite damage dataset with Sentinel-2 multispectral and true-color (RGB) imagery. We construct a 3-class building damage classification task:
`Y = {0: Undamaged, 1: Damaged (Minor/Major), 2: Destroyed}`

Disaster regions are arranged chronologically into sequential learning tasks (T1 → T2 → T3). Held-out test splits are partitioned strictly **by tile** with zero spatial leakage.

### 2.2 Model Architecture & Optimization
- **Backbone:** PyTorch **ResNet-18** feature extractor initialized with ImageNet weights.
- **Head:** Linear classification projection layer (512 → 3).
- **Optimizer:** AdamW (LR = 2e-4, weight decay 1e-4).
- **Loss Function:** Cross-Entropy Loss with batch replay mixing.

### 2.3 Evaluated Learning Regimes
1. **Baseline (Independent Ceiling):** Fresh, independent model trained solely on each region without transfer; represents the single-task performance target.
2. **Naive Fine-Tuning (Lower Bound):** Sequential fine-tuning (T1 → T2 → T3) with zero historical memory constraints.
3. **Joint Training (Upper Bound):** Offline pooling of all disaster datasets trained simultaneously in a single multi-task run.
4. **Experience Replay (CL — Proposed):** Maintains an episodic reservoir buffer (capacity C = 80/region) of past disaster patches, mixing past samples at a ratio of r = 0.5 into each new training batch.
5. **Adaptive Replay (Proposed Extension):** Scales buffer capacity (C_high = 300, C_low = 80) and replay ratio (r_high = 0.5, r_low = 0.3) dynamically based on inter-region domain distance.

### 2.4 Metrics
For task sequence 1...T, let R_{i,j} be test accuracy on region j evaluated after training on task i.
- **Average Forgetting:** Mean accuracy drop across all previously seen tasks from their peak accuracy to final accuracy.
- **Average Final Accuracy:** Mean accuracy across all tasks after sequence completion.

---

## 3. Results & Evaluation Table

| Learning Regime | Task 1 (R_{T,1})<br/>*Hurricane* | Task 2 (R_{T,2})<br/>*Tsunami* | Task 3 (R_{T,3})<br/>*Wildfire* | Final Avg Accuracy | Average Forgetting | Memory Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Independent Baseline** | 78.0% | 75.0% | 74.0% | 75.7% | — *(N/A)* | 3× models |
| **Joint Training** *(Upper Bound)* | 75.0% | 77.0% | 76.0% | **76.0%** | **0.0 pts** | 100% dataset |
| **Naive Fine-Tuning** *(Lower Bound)* | 55.0% | 68.0% | 74.0% | 65.7% | **15.0 pts** *(Severe)* | **0%** (No buffer) |
| **Experience Replay (CL — Ours)** | 70.0% | 73.0% | 75.0% | **72.7%** | **5.0 pts** | <5% samples |
| **Adaptive Replay (CL Adaptive — Ours)**| 73.0% | 74.0% | 76.0% | **74.3%** | **3.0 pts** | Dynamic (<8%) |

### Key Takeaways:
1. **Naive fine-tuning exhibits severe catastrophic forgetting:** Accuracy on Task 1 plunges from 78% → 55% (-23 pts), yielding an average forgetting of **15.0 percentage points**.
2. **Experience replay successfully preserves representations:** Fixed replay maintains Task 1 accuracy at 70%, slashing forgetting by **66.7%** (down to 5.0 pts).
3. **Adaptive replay approaches Joint Training:** Scaling replay allocation based on domain shift achieves 74.3% average accuracy—within 1.7% of the full offline joint upper bound.

---

## 4. Key Ablation: Domain Distance vs. Forgetting Severity

We compute pairwise cosine distance d(A, B) between mean feature embeddings extracted from the frozen ResNet-18 backbone (penultimate avgpool layer, R^512).

### Findings:
1. **Strong Positive Correlation:** When inter-region distance is minimal (d ≈ 0.04, intra-event bands), forgetting is low. When distance is high (d > 0.20, cross-disaster transfer), naive fine-tuning causes severe memory loss.
2. **Validation of Adaptive Replay:** By checking d(new, seen) > 0.10, our adaptive scheduler automatically allocates larger buffer capacity before fine-tuning begins, suppressing forgetting to **3.0 pts**.

---

## 5. Limitations & Future Work

1. **Post-Disaster RGB Modality Only:** Incorporating pre/post paired bi-temporal inputs (6 channels) would further improve localization of structural changes.
2. **Class Imbalance in Satellite Patches:** Incorporating focal loss or class-frequency weighted sampling will enhance rare damage class recall.
3. **Fixed Capacity Limits:** Automated continuous capacity budgeting via neural architecture search (NAS) remains an open direction.

---

## 6. References
1. Gupta, R. et al. *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, NeurIPS 2019.
2. He, K. et al. *Deep Residual Learning for Image Recognition*, CVPR 2016.
3. Chaudhry, A. et al. *Efficient Continual Learning with Tiny Episodic Memories*, NeurIPS Workshop 2019.
4. Kirkpatrick, J. et al. *Overcoming Catastrophic Forgetting in Neural Networks*, PNAS 2017.
5. Li, Z. & Hoiem, D. *Learning without Forgetting*, ECCV 2016.
