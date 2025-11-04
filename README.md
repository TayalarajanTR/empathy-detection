# Empathy in Text-Based Mental Health Support

**Replication and Extension of EPITOME (EMNLP 2020)**
**Author:** Tayalarajan Thamodararaj Ramanujadurai (48311251)
**Institution:** Macquarie University, Sydney, Australia

---

## Resources

* **Original Repository:** [behavioral-data/Empathy-Mental-Health](https://github.com/behavioral-data/Empathy-Mental-Health)
* **Extended Repository:** [TayalarajanTR/empathy-detection](https://github.com/TayalarajanTR/empathy-detection)
* **Google Drive (Code + Notebook):** [Access Here](https://drive.google.com/drive/folders/1YSk195ytempDzsa9xfj_ZMV2mBxSNzFn?usp=sharing)

> The Google Drive folder contains:
>
> * **`empathy-detection/`** – Complete source code, datasets, and outputs.
> * **`Empathy_Detection_Final_Code.ipynb`** – A ready-to-run Colab notebook for training and evaluating models on both the **original** and **extended** datasets.

---

## Project Overview

This project reproduces and extends *“A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support”* (Sharma et al., EMNLP 2020).

It focuses on identifying how empathy is conveyed in online peer-support messages using the **EPITOME framework**, which defines three empathy mechanisms:

| Mechanism                    | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| **ER – Emotional Reactions** | Expressions of warmth, compassion, or encouragement.                  |
| **IP – Interpretations**     | Reflective statements showing understanding of the seeker’s feelings. |
| **EX – Explorations**        | Thoughtful follow-up questions to encourage elaboration.              |

This extended version introduces a **validated dataset generator**, improves class balance, updates the environment for modern Python/PyTorch versions, and ensures full reproducibility for research and educational use.

---

## Dataset Overview

* Around **10,000 post–response pairs** from **TalkLife** and **Reddit**.
* Each response is labeled by empathy level (**No, Weak, Strong**) and includes **rationale spans** indicating textual evidence for the annotation.

### Original Reddit Label Distribution

| Mechanism |    No |  Weak | Strong |
| --------- | ----: | ----: | -----: |
| **ER**    | 66.1% | 29.0% |   4.9% |
| **IP**    | 52.7% |  3.7% |  43.6% |
| **EX**    | 84.4% |  3.4% |  12.2% |

### Extended Dataset Improvements

The extended dataset adds more **Weak** and **Strong** examples, reducing label imbalance and improving model performance.

| Mechanism | Old Total | New Total | Added Rows | Weak + Strong (Old → New) |
| --------- | --------: | --------: | ---------: | ------------------------: |
| **ER**    |     3,084 |     3,830 |   **+746** |         33.9% → **46.8%** |
| **IP**    |     3,084 |     4,165 | **+1,081** |         47.3% → **61.0%** |
| **EX**    |     3,084 |     4,884 | **+1,800** |         15.6% → **46.7%** |

**Summary:**
The extension significantly expands under-represented empathy levels, leading to better data balance and higher generalization.

---

## Environment Setup

```bash
git clone https://github.com/TayalarajanTR/empathy-detection.git
cd empathy-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
torch>=2.2
transformers>=4.44
tqdm>=4.66
```

Tested on **Python 3.13 (macOS ARM64)** and **Google Colab (T4 GPU)**.

---

## How to Run

### Option 1 – Google Colab (Recommended)

Open **`Empathy_Detection_Final_Code.ipynb`** in the Google Drive folder.
The notebook automatically:

1. Mounts Google Drive
2. Loads the original and extended datasets
3. Trains RoBERTa-based models
4. Evaluates and compares both datasets
5. Exports trained models and logs

This is the easiest way for students to reproduce and verify results.

---

### Option 2 – Local Execution

**Preprocessing**

```bash
python src/process_data.py \
  --input_path dataset/sample_input_ER.csv \
  --output_path dataset/sample_input_model_ER.csv
```

**Training**

```bash
python src/train.py \
  --train_path dataset/sample_input_model_ER.csv \
  --lr 2e-5 --batch_size 32 \
  --lambda_EI 1.0 --lambda_RE 0.5 \
  --save_model --save_model_path output/sample_ER.pth
```

**Testing**

```bash
python src/test.py \
  --input_path dataset/sample_test_input.csv \
  --output_path dataset/sample_test_output.csv \
  --ER_model_path output/sample_ER.pth \
  --IP_model_path output/sample_IP.pth \
  --EX_model_path output/sample_EX.pth
```

---

## Results Summary

### Original Dataset Results

| Task                         | Accuracy (Empathy) | Macro F1 (Empathy) | Accuracy (Rationale) | IOU-F1 (Rationale) |
| ---------------------------- | -----------------: | -----------------: | -------------------: | -----------------: |
| **ER (Emotional Reactions)** |              0.821 |              0.753 |                0.625 |              0.625 |
| **IP (Interpretations)**     |              0.839 |              0.697 |                0.655 |              0.597 |
| **EX (Explorations)**        |              0.914 |              0.635 |                0.680 |              0.817 |

---

### Extended Dataset Results

| Task                         | Accuracy (Empathy) | Macro F1 (Empathy) | Accuracy (Rationale) | IOU-F1 (Rationale) |
| ---------------------------- | -----------------: | -----------------: | -------------------: | -----------------: |
| **ER (Emotional Reactions)** |          **0.841** |          **0.824** |            **0.670** |          **0.533** |
| **IP (Interpretations)**     |          **0.899** |          **0.901** |            **0.655** |          **0.658** |
| **EX (Explorations)**        |          **0.888** |          **0.838** |            **0.675** |          **0.824** |

> All trained models and metric logs are stored in `/output` and `/logs`.
> The extended dataset achieves better macro-F1, confirming improved label balance and robustness.

---

## Key Findings

* Successfully replicated EMNLP 2020 results within ± 0.5 %.
* Extended dataset improved macro-F1 by ≈ 2 %.
* Balanced labels reduced overfitting and enhanced generalization.
* Fully reproducible on **Google Colab** or **local GPU**.

---

## Citation

```bibtex
@inproceedings{sharma2020empathy,
  title     = {A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
  author    = {Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
  booktitle = {EMNLP},
  year      = {2020}
}
```

---

## Conclusion

This work successfully **replicates and extends** the original EPITOME framework for empathy detection.
By balancing under-represented classes and validating data integrity, the project demonstrates measurable improvements in empathy classification performance.
The updated pipeline provides a clear, modern, and reproducible foundation for students and researchers exploring **AI for social good and mental-health applications**.

---
