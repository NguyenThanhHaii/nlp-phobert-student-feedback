# Vietnamese Student Feedback NLP

**Sentiment Analysis and Topic Classification of Noisy Vietnamese Student Feedback using PhoBERT**

Dự án này thực hiện phân tích cảm xúc và phân loại chủ đề phản hồi sinh viên tiếng Việt, tập trung vào ảnh hưởng của dữ liệu phi chuẩn như bỏ dấu, teencode, viết tắt, lỗi gõ và kéo dài ký tự. Dự án so sánh baseline TF-IDF/SVM với PhoBERT, sau đó đánh giá độ bền của mô hình trên các biến thể nhiễu có kiểm soát.

> Demo chỉ phục vụ minh họa học thuật. Kết quả không phải công cụ đánh giá chính thức về chất lượng giảng dạy hoặc quyết định hành chính.

---

## 1. Problem Statement

Phản hồi sinh viên tiếng Việt thường ngắn, không đồng đều về cách viết và có thể chứa nhiều dạng phi chuẩn:

- thiếu dấu tiếng Việt;
- viết tắt như `gv`, `sv`, `csvc`, `ctdt`;
- teencode như `ko`, `k`, `đc`, `lun`;
- lỗi gõ;
- kéo dài ký tự như `hayyy`, `tốttt`.

Các hiện tượng này có thể làm thay đổi biểu diễn từ/subword, ảnh hưởng đến các mô hình phân loại văn bản. Dự án đặt ra hai bài toán chính:

1. **Sentiment classification**: phân loại phản hồi thành `negative`, `neutral`, `positive`.
2. **Topic classification**: phân loại phản hồi thành `lecturer`, `training_program`, `facility`, `others`.

---

## 2. Dataset

Dataset sử dụng:

```text
uitnlp/vietnamese_students_feedback
```

Schema đã xác minh trong Stage 1:

```text
sentence  -> text
sentiment -> sentiment_label
topic     -> topic_label
```

Label mapping:

```text
Sentiment:
0 -> negative
1 -> neutral
2 -> positive

Topic:
0 -> lecturer
1 -> training_program
2 -> facility
3 -> others
```

Splits:

```text
train: 11426
dev:   1583
test:  3166
```

Dataset có mất cân bằng lớp. Với sentiment, lớp `neutral` nhỏ và khó hơn hai lớp còn lại. Với topic, lớp `lecturer` chiếm đa số, trong khi `facility` và `others` là các lớp nhỏ hơn.

---

## 3. Project Structure

```text
nlp-phobert-student-feedback/
├── app/
│   ├── backend/
│   └── frontend/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── noisy/
│   ├── normalized/
│   └── audit/
├── models/
│   ├── baselines/
│   │   ├── sentiment/
│   │   └── topic/
│   └── phobert/
│       ├── sentiment/
│       └── topic/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── notes/
│   └── tables/
├── scripts/
├── src/
├── requirements.txt
├── requirements_demo.txt
└── README.md
```

---

## 4. Methodology

### 4.1 Baseline models

Baseline sử dụng các mô hình truyền thống:

- Majority Class baseline;
- Word TF-IDF + Linear SVM;
- Character TF-IDF + Linear SVM.

Metric chính:

```text
Macro-F1
```

Macro-F1 được chọn vì dataset mất cân bằng lớp và cần đánh giá công bằng hơn cho các lớp nhỏ như `neutral`, `facility`, `others`.

### 4.2 PhoBERT fine-tuning

PhoBERT được fine-tune riêng cho hai task:

```text
sentiment classification
topic classification
```

Cấu hình chính:

```text
model: vinai/phobert-base
max_length: 128
learning_rate: 2e-5
epochs: 4
batch_size: 16
primary metric: Macro-F1
```

Fine-tuning được chạy trên Kaggle GPU. Trong implementation hiện tại, external word segmentation không được bật; tokenizer dùng theo cấu hình checkpoint PhoBERT.

### 4.3 Controlled noisy test generation

Dự án không train trên noisy data. Thay vào đó, noisy test sets được tạo từ test set sạch để đánh giá robustness.

Các loại nhiễu:

```text
no_accent
domain_abbreviation
teencode_colloquial
typo
elongation
mixed_noise
```

### 4.4 Segmentation and tokenization analysis

Stage 5 phân tích ảnh hưởng của noise lên:

- số segment theo whitespace fallback;
- số subword theo PhoBERT tokenizer;
- subword inflation ratio;
- tokenization shift.

### 4.5 Clean vs noisy evaluation

Stage 6 đánh giá baseline và PhoBERT trên:

```text
clean test
noisy test sets
changed-only subsets
```

### 4.6 Error analysis

Stage 7 phân tích:

- error rate theo task/model/noise;
- confusion pairs;
- lỗi lớp nhỏ/dễ nhầm;
- lỗi riêng với no-accent;
- liên hệ giữa subword inflation và error rate.

### 4.7 Optional normalization

Stage 8 kiểm tra normalization rule-based đơn giản:

- giảm ký tự kéo dài;
- mở rộng viết tắt;
- mở rộng teencode;
- khôi phục một số cụm không dấu phổ biến;
- sửa một số lỗi gõ đơn giản.

Thí nghiệm này chỉ là hướng hỗ trợ, không thay thế kết quả chính.

---

## 5. Main Results

### 5.1 Clean test performance

| Task      | Model           | Accuracy | Macro-F1 | Weighted-F1 |
| --------- | --------------- | -------: | -------: | ----------: |
| Sentiment | TF-IDF char SVM |   0.8740 |   0.7354 |      0.8755 |
| Sentiment | TF-IDF word SVM |   0.8920 |   0.7289 |      0.8870 |
| Sentiment | PhoBERT-base    |   0.9324 |   0.8295 |      0.9312 |
| Topic     | TF-IDF char SVM |   0.8326 |   0.7299 |      0.8396 |
| Topic     | TF-IDF word SVM |   0.8585 |   0.7509 |      0.8598 |
| Topic     | PhoBERT-base    |   0.8992 |   0.8052 |      0.8969 |

PhoBERT đạt hiệu năng tốt nhất trên clean test ở cả hai task.

### 5.2 PhoBERT under noisy test sets

| Task      | Noise type          | Macro-F1 |
| --------- | ------------------- | -------: |
| Sentiment | clean               |   0.8295 |
| Sentiment | no_accent           |   0.3165 |
| Sentiment | domain_abbreviation |   0.8257 |
| Sentiment | teencode_colloquial |   0.8058 |
| Sentiment | typo                |   0.7848 |
| Sentiment | elongation          |   0.7995 |
| Sentiment | mixed_noise         |   0.7708 |
| Topic     | clean               |   0.8052 |
| Topic     | no_accent           |   0.2630 |
| Topic     | domain_abbreviation |   0.7417 |
| Topic     | teencode_colloquial |   0.7941 |
| Topic     | typo                |   0.7645 |
| Topic     | elongation          |   0.7798 |
| Topic     | mixed_noise         |   0.7099 |

`no_accent` là noise gây suy giảm mạnh nhất. Kết quả này phù hợp với phân tích tokenization: bỏ dấu làm thay đổi mạnh biểu diễn subword của văn bản tiếng Việt.

### 5.3 Normalization experiment

Normalization rule-based phục hồi rõ nhất ở `no_accent`.

| Task      | Model           | Noise     | Macro-F1 noisy | Macro-F1 normalized | Improvement |
| --------- | --------------- | --------- | -------------: | ------------------: | ----------: |
| Sentiment | TF-IDF char SVM | no_accent |         0.4050 |              0.5325 |     +0.1275 |
| Sentiment | TF-IDF word SVM | no_accent |         0.3597 |              0.5505 |     +0.1908 |
| Topic     | TF-IDF char SVM | no_accent |         0.2875 |              0.4716 |     +0.1842 |
| Topic     | TF-IDF word SVM | no_accent |         0.2614 |              0.4900 |     +0.2286 |

Normalization có ích trong một số trường hợp, nhưng không ổn định tuyệt đối và có thể làm giảm nhẹ Macro-F1 với một vài noise type.

---

## 6. Key Findings

1. **PhoBERT đạt hiệu năng tốt nhất trên clean test.**

2. **No-accent là loại nhiễu nghiêm trọng nhất.**  
   Cả baseline và PhoBERT đều giảm mạnh khi dữ liệu bị bỏ dấu.

3. **PhoBERT tốt hơn baseline trong phần lớn điều kiện noisy, nhưng không miễn nhiễm với nhiễu tiếng Việt phi chuẩn.**

4. **Character TF-IDF/SVM có độ bền tương đối tốt trong một số tình huống nhiễu nhẹ**, đặc biệt khi so sánh theo mức suy giảm tuyệt đối.

5. **Các lớp nhỏ như `neutral`, `facility`, `others` vẫn là điểm yếu.**  
   Điều này liên quan trực tiếp đến mất cân bằng lớp trong dataset.

6. **Normalization có thể phục hồi một phần hiệu năng**, đặc biệt với no-accent, nhưng không phải giải pháp robust hoàn chỉnh.

---

## 7. Demo App

Demo web nằm trong:

```text
app/backend/
app/frontend/
```

### 7.1 Install demo requirements

```powershell
python -m pip install -r requirements_demo.txt
```

### 7.2 Run demo

```powershell
.\scripts\run_demo.ps1
```

Open:

```text
http://127.0.0.1:8000
```

### 7.3 API endpoints

```text
GET  /health
GET  /model-info
POST /predict
```

Example request:

```json
{
  "text": "gv day de hieu nhung bt nhieu qua",
  "task": "both",
  "model": "auto",
  "normalize": true
}
```

### 7.4 Model behavior

```text
model = auto
```

The backend tries to use PhoBERT if checkpoint files are available. If PhoBERT is unavailable, it falls back to baseline TF-IDF/SVM.

Default baseline models:

```text
sentiment: tfidf_char_svm
topic: tfidf_word_svm
```

PhoBERT checkpoints should be placed at:

```text
models/phobert/sentiment/best_model/
models/phobert/topic/best_model/
```

---

## 8. Reproducibility

### 8.1 Environment

Recommended:

```text
Python 3.10
Conda environment: nlp-phobert-feedback
```

Install main dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install demo dependencies:

```powershell
python -m pip install -r requirements_demo.txt
```

### 8.2 Notebook order

Run notebooks in this order:

```text
01_dataset_verification_eda.ipynb
02_baseline_models.ipynb
03_phobert_clean_finetuning.ipynb
04_noise_generation_audit.ipynb
05_segmentation_tokenization_analysis.ipynb
06_noisy_evaluation_all_models.ipynb
06b_kaggle_phobert_noisy_evaluation_v2.ipynb
07_error_analysis.ipynb
08_optional_normalization.ipynb
```

Notes:

- Stage 3 and Stage 6B are recommended on Kaggle GPU.
- Stage 6A baseline evaluation can run locally.
- Stage 8 local version evaluates baseline normalization only.

---

## 9. Important Outputs

### Reports

```text
reports/notes/03_phobert_clean_finetuning_report.md
reports/notes/04_noise_generation_audit_report.md
reports/notes/05_segmentation_tokenization_report.md
reports/notes/06_noisy_evaluation_report.md
reports/notes/06b_phobert_noisy_evaluation_report.md
reports/notes/07_error_analysis_report.md
reports/notes/08_optional_normalization_report.md
reports/notes/09_demo_guide.md
```

### Summary tables

```text
reports/tables/02_baseline_results_summary.csv
reports/tables/03_phobert_results_summary.csv
reports/tables/03_phobert_vs_baseline_comparison.csv
reports/tables/04_noise_generation_summary.csv
reports/tables/05_segmentation_tokenization_summary.csv
reports/tables/06_full_model_robustness_ranking.csv
reports/tables/07_error_summary_by_model_noise_task.csv
reports/tables/08_normalization_improvement_summary.csv
```

### Figures

```text
reports/figures/
```

---

## 10. Git and Large Files

Do not commit large generated prediction files or model checkpoints unless explicitly required.

Usually do **not** commit:

```text
models/phobert/
reports/tables/*predictions*.csv
reports/tables/07_predictions_with_text_and_tokenization.csv
data/noisy/*.csv
data/normalized/*.csv
```

Recommended to commit:

```text
configs/
src/
notebooks/
app/
scripts/
reports/notes/
selected reports/tables/*.csv
selected reports/figures/*.png
README.md
```

---

## 11. Limitations

- Dataset is domain-specific: Vietnamese student feedback.
- Noisy data is rule-generated, not fully human-validated.
- Normalization rules are simple and incomplete.
- PhoBERT normalization evaluation is not included in the local Stage 8 run.
- The demo is for academic presentation only.
- The system is not a production moderation or education-quality assessment tool.

---

## 12. Project Status

```text
Stage 0  Project setup                         DONE
Stage 1  Dataset verification and EDA           DONE
Stage 2  Baseline models                        DONE
Stage 3  PhoBERT clean fine-tuning              DONE
Stage 4  Controlled noisy test generation       DONE
Stage 5  Segmentation/tokenization analysis     DONE
Stage 6  Clean vs noisy evaluation              DONE
Stage 7  Error analysis                         DONE
Stage 8  Optional normalization experiment      DONE
Stage 9  Demo app                               DONE
```

---

## 13. Academic Disclaimer

This project is developed for academic research and demonstration. It should not be used as an official tool for evaluating lecturers, students, courses, or university operations.
