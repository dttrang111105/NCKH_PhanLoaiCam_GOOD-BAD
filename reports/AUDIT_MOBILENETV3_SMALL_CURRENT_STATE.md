# Báo cáo audit MobileNetV3-Small — hiện trạng project

**Ngày audit:** 2026-09-30  
**Phạm vi:** `E:\orange_project`  
**Chế độ:** chỉ đọc và kiểm tra nhẹ; không sửa source, không cài package, không train lại, không chạy lại test cuối, không điều khiển camera/phần cứng. File này là thay đổi duy nhất của audit.

## 1. Tóm tắt hiện trạng

Project hiện có hai phần:

1. **Baseline image classification:** MobileNetV3-Small phân loại một crop/ảnh cam thành `GOOD=0` hoặc `BAD=1`.
2. **Demo camera kết hợp:** OpenCV dùng ngưỡng màu HSV + contour để tìm vùng giống quả cam, sau đó model classification chấm từng crop. Đây **không phải AI object detector**.

### ĐÃ XÁC MINH

- Pipeline training dùng TensorFlow/Keras MobileNetV3Small, `weights="imagenet"`, `include_preprocessing=True`, input RGB `224×224`, head `GlobalAveragePooling2D → Dropout(0.2) → Dense(1, sigmoid)`, Binary Cross-Entropy.
- Dataset manifest có 15.000 ảnh: mỗi lớp gốc 3.000 ảnh; chia train/validation/test = 10.500/2.250/2.250, stratify theo 5 lớp gốc với seed 42.
- Có Keras checkpoint load được: 939.697 tham số, input `(None, 224, 224, 3)`, output `(None, 1)` sigmoid.
- Kết quả internal test đã lưu: accuracy 99,9111%, Recall BAD 100%, 0 BAD→GOOD và 2 GOOD→BAD trên 2.250 ảnh.
- Có artifact TFLite full INT8 1.216.992 byte; input `int8 [1,224,224,3]`, scale 1.0, zero-point -128; output `int8 [1,1]`, scale 1/256, zero-point -128. Một lần invoke với tensor zero chạy thành công trên máy audit.
- Source compile thành công và môi trường runtime không có dependency bị vỡ theo `pip check`.

### SUY LUẬN TỪ CODE, CHƯA CHẠY

- Augmentation chỉ áp dụng cho train; validation/test không có random augmentation.
- Fine-tune chỉ mở tối đa 30 layer cuối của backbone và giữ BatchNorm frozen.
- Runtime đổi BGR→RGB và giữ pixel float trong miền 0–255, phù hợp với preprocessing nằm trong MobileNetV3Small.
- Representative set cho INT8 quét đệ quy toàn bộ path `E:/dataset`; theo cấu hình hiện tại path này chứa cả train/validation/test, nên calibration có nguy cơ nhìn thấy phân phối test.

### CHƯA CÓ BẰNG CHỨNG

- Split độc lập theo **từng quả**, video hay session chụp; manifest không có `fruit_id`, `video_id`, `session_id`.
- Không có bằng chứng ảnh trùng/ảnh gần trùng không đi qua nhiều split. Audit không kết luận leakage đã xảy ra.
- Không có external/camera report, stress test ảnh khó, test theo quả, nhiều seed/run, confidence interval hay calibration metric.
- Không có kết quả parity Keras FP32 ↔ TFLite INT8 trên validation/test.
- Không có benchmark Raspberry Pi 4, p50/p95, RAM, nhiệt độ, threads, warm-up hay end-to-end camera→servo.
- Chưa có Multi-Exit, KD, Confidence/Entropy baseline, EENet hoặc QRT-Exit.

**Kết luận ngắn:** project đã có một baseline classification nội bộ rất mạnh và artifact INT8 thật, nhưng baseline **chưa đủ đáng tin để khóa** do chưa chứng minh split độc lập theo quả/session và chưa có external/camera evaluation. Kết quả 99,91% hiện chỉ là metric theo ảnh của internal split.

## 2. Bảng thành phần

| Thành phần | Hiện trạng | Bằng chứng | Thiếu gì |
|---|---|---|---|
| Dataset manifest | 15.000 ảnh, 5 lớp gốc, 3 split | `training/outputs/splits/dataset_manifest.csv`; `config.yaml:8-20` | ID quả/video/session; audit duplicate/near-duplicate |
| Label mapping | `fresh_orange→GOOD=0`; 4 lớp còn lại `BAD=1` | `training/config.yaml:16-20`; `training/orange/data.py:21-33` | Tài liệu quy tắc nhãn/kiểm tra chất lượng nhãn |
| Split | 70/15/15, seed 42, stratify theo lớp gốc | `training/config.yaml:2,10-12`; `training/orange/data.py:139-182` | Group-aware split và báo cáo leakage |
| Augmentation | Có, chỉ train | `training/orange/data.py:273-296` | Ablation/ghi lại augmentation trong run metadata |
| Backbone | MobileNetV3Small Keras Applications | `training/orange/model.py:15-35` | Hash/metadata xác nhận nguồn init của từng run |
| Training | Two-stage head + fine-tune | `training/orange/model.py:116-184`; training log | Run ID, config snapshot, environment snapshot, Git commit |
| Threshold | 0,14, chọn trên validation | `selected_threshold.json`; `evaluation.py:86-120` | Quy tắc tie-break rõ hơn; calibration sau INT8 |
| Internal evaluation | Có đủ metric và plots theo ảnh | `outputs/reports/internal_test/*` | Group/fruit-level metric, confidence interval, multi-seed |
| External evaluation | Code và config có; artifact không có | `training/main.py:246-320`; chỉ có report `validation` và `internal_test` | External/camera holdout report |
| Keras artifacts | `best_model.keras`, `final_model.keras` đều có | `training/outputs/models/` | Manifest liên kết artifact–config–data–commit |
| TFLite INT8 | Artifact có và load/invoke được | `model.tflite`; `convert_to_tflite.py:83-151` | Accuracy/Recall BAD/parity/latency đúng quy trình |
| Camera demo | HSV contour + classifier, camera 0 | `app.py:65-301` | Dataset detector; end-to-end recall; camera CLI; Pi validation |
| Raspberry Pi | Mới là hướng triển khai | README; không có benchmark artifact | Runtime LiteRT/TFLite thật và benchmark Pi 4 |
| Early Exit/QRT | CHƯA CÓ | Không có source/artifact tương ứng | Chỉ bắt đầu sau khi khóa baseline |
| Test tự động | CHƯA CÓ | Không có file test/notebook/CI | Unit/integration/regression tests |
| Git/provenance | Thư mục không phải Git repository | `git status` trả `fatal: not a git repository` | Branch, commit, diff, tag run |

## 3. Cây thư mục, entry point và luồng xử lý

```text
orange_project/
├── app.py                         # runtime camera, HSV detector + classifier
├── convert_to_tflite.py           # Keras → full INT8 TFLite
├── model.tflite                  # artifact INT8, ~1,16 MiB
├── threshold.json                # threshold runtime = 0,14
├── requirements.txt              # dependency runtime Windows
├── README.md
├── venv/                         # virtualenv runtime cục bộ
└── training/
    ├── main.py                     # CLI prepare/train/evaluate/predict
    ├── config.yaml
    ├── requirements.txt
    ├── orange/
    │   ├── data.py
    │   ├── model.py
    │   └── evaluation.py
    └── outputs/                    # split, log, Keras model, metric, plots
```

Entry point theo code:

- `cd training && python main.py prepare|train|evaluate-all|run|predict`
- `python convert_to_tflite.py --dataset <representative-dir>`
- `python app.py --camera`

Luồng training hiện tại:

```text
E:/dataset/{5 class folders}
  → validate ảnh + manifest
  → random stratified split theo original_class (seed 42)
  → decode RGB + resize 224×224
  → augmentation chỉ train
  → MobileNetV3Small ImageNet preprocessing nội bộ
  → train head → fine-tune 30 layer cuối
  → checkpoint theo validation AUC
  → chọn threshold theo validation F1 BAD
  → internal test theo từng ảnh
  → best_model.keras
  → representative calibration → model.tflite INT8
```

Luồng camera hiện tại:

```text
camera 0 → HSV mask → morphology/contour/shape filters → crop
         → BGR→RGB → resize 224×224 → INT8 quantize
         → TFLite sigmoid probability BAD → threshold 0,14
         → GOOD/BAD + bounding box
```

Do detector HSV đứng trước classifier, metric classifier không đo được quả cam bị detector bỏ sót.

## 4. Framework, dependency và cấu hình training

### Phiên bản khai báo

- Runtime: TensorFlow 2.16.1, NumPy 1.26.4, OpenCV 4.10.0.84 (`requirements.txt`).
- Training: thêm pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.9.1, Pillow 10.4.0, PyYAML 6.0.1, tqdm 4.66.4 (`training/requirements.txt`).

### Phiên bản đã xác minh trong `venv`

- Python 3.11.9; TensorFlow 2.16.1; NumPy 1.26.4; OpenCV 4.10.0.
- `venv` hiện tại không có pandas, sklearn, matplotlib, Pillow, PyYAML, tqdm, nên đây là runtime env chứ không phải training env đầy đủ.

### Cấu hình baseline

| Hạng mục | Giá trị | Bằng chứng |
|---|---:|---|
| Seed | 42 | `training/config.yaml:2`; `data.py:35-45` |
| Input | RGB 224×224×3, float32 0–255 trước model | `config.yaml:9`; `data.py:260-266` |
| Backbone | `tf.keras.applications.MobileNetV3Small` | `model.py:20-25` |
| Pretrained config | `weights="imagenet"` | `model.py:23` |
| Top | Không dùng ImageNet top | `model.py:22` |
| Preprocessing | Nằm trong backbone | `model.py:24` |
| Classifier | GAP + Dropout 0,20 + Dense(1, sigmoid) | `model.py:30-34`; config line 29 |
| Loss | BinaryCrossentropy | `model.py:42-50` |
| Optimizer | Adam | `model.py:43` |
| Head phase | 12 epoch, LR 1e-3, backbone frozen | `config.yaml:24,26`; `model.py:26,143-151` |
| Fine-tune | 18 epoch tối đa, LR 1e-5, 30 layer cuối | `config.yaml:25,27-28`; `model.py:54-73,153-172` |
| BatchNorm | Giữ frozen; backbone được gọi `training=False` | `model.py:28-29,63-66` |
| Batch size | 16 | `config.yaml:23` |
| Scheduler | ReduceLROnPlateau factor 0,3, patience 2, min LR 1e-7 | `model.py:103-110`; config line 31 |
| Early stopping | Theo `val_auc`, patience 5, không restore trong-memory | `model.py:96-102` |
| Best checkpoint | `val_auc`, mode max, validation split | `model.py:87-93` |
| Imbalance | Class weight theo split train | `data.py:229-238` |
| Weight suy ra | GOOD 2,5; BAD 0,625 | Từ train: 2.100 GOOD, 8.400 BAD |
| Threshold search | 0,10–0,90, step 0,01 | `config.yaml:34-37` |
| Threshold objective | F1 BAD → Recall BAD → Accuracy → FN | `evaluation.py:86-120` |

Log có 12 epoch head và 9 epoch fine-tune. Số epoch reset về 0 giữa hai phase và CSV không có cột `phase`, `run_id`, config hash hay data hash. Checkpoint theo validation, không theo test. Test được gọi sau khi checkpoint và threshold đã chọn trên validation (`training/main.py:193-223`), nên **không thấy code dùng test để chọn checkpoint/threshold**.

## 5. Dataset và rò rỉ dữ liệu

### Phân bố đã xác minh

| Lớp gốc | Nhãn | Train | Validation | Test | Tổng |
|---|---|---:|---:|---:|---:|
| fresh_orange | GOOD=0 | 2.100 | 450 | 450 | 3.000 |
| blackspotted_orange | BAD=1 | 2.100 | 450 | 450 | 3.000 |
| cankered_orange | BAD=1 | 2.100 | 450 | 450 | 3.000 |
| green_orange | BAD=1 | 2.100 | 450 | 450 | 3.000 |
| rotten_orange | BAD=1 | 2.100 | 450 | 450 | 3.000 |
| **Tổng** | 20% GOOD / 80% BAD | **10.500** | **2.250** | **2.250** | **15.000** |

`invalid_images.csv` có 0 dòng dữ liệu. Các path trong config `E:/dataset` và external `E:/orange_new_dataset/test` tồn tại tại thời điểm audit, nhưng external report không có.

### Kiểm tra tính độc lập

- Manifest chỉ có `image_path,file_name,original_class,label,binary_label,split`; không có identity/group metadata.
- Split dùng `train_test_split(... stratify=original_class, random_state=seed)` trên từng file (`data.py:161-176`).
- Tên file dạng `fresh_orange_0000.jpg` không cho phép suy ra chắc chắn ID quả/session.
- Không có code pHash/perceptual duplicate/group split trong pipeline hiện tại.
- Kiểm tra hash toàn bộ dataset đã được dừng vì vượt giới hạn kiểm tra nhẹ; không có kết quả hash để báo cáo.

Vì vậy, **chưa thể chứng minh split độc lập theo quả**. Đây là rủi ro, không phải kết luận leakage đã xảy ra.

Augmentation gồm horizontal flip, rotation 0,04, zoom 0,10, translation 0,05 và contrast 0,10; nó nằm bên trong nhánh `if training`, sau split, nên không ghi augmentation thành file sang validation/test (`data.py:273-296`).

## 6. Kết quả baseline đã xác minh

Nguồn: `training/outputs/reports/internal_test/metrics.json`, `classification_report.txt`, `per_original_class.csv`, `confusion_matrix.png`.

| Metric | Internal test Keras | Phạm vi/mẫu số |
|---|---:|---|
| Số ảnh | 2.250 | Theo ảnh |
| Threshold | 0,14 | Chọn trên validation |
| Accuracy | 0,999111 (99,9111%) | 2.248/2.250 đúng |
| Precision BAD | 0,998890 (99,8890%) | 1.800/(1.800+2) |
| Recall BAD | 1,000000 (100%) | 1.800/1.800 BAD |
| F1 BAD | 0,999445 (99,9445%) | Theo ảnh |
| Macro F1 | 0,998609 | GOOD/BAD |
| ROC AUC | 1,000000 | Theo probability |
| BAD→GOOD | 0 | 0/1.800 = 0% |
| GOOD→BAD | 2 | 2/450 = 0,4444% |
| Recall GOOD | 0,995556 | 448/450 |
| Avg inference | 3,2905 ms/ảnh | Artifact không ghi device, warm-up, batch/timing protocol |

Confusion matrix, với hàng là nhãn thật `[GOOD, BAD]` và cột là dự đoán `[GOOD, BAD]`:

```text
[[ 448,    2],
 [   0, 1800]]
```

BAD được xử lý đúng là positive class: prediction là `probability >= threshold`, `labels=[0,1]`, và code lấy index 1 cho Precision/Recall/F1 BAD (`evaluation.py:42-81`). Recall BAD cao không đến từ việc dự đoán mọi ảnh là BAD, vì chỉ 2/450 GOOD bị loại nhầm.

Validation tại threshold 0,14 có accuracy 99,9556%, Precision BAD 99,9445%, Recall BAD 100%, 0 FN BAD và 1 FP GOOD. Nhiều threshold cùng metric tối ưu; code không quy đị tie-break theo margin/threshold cao hơn, nên 0,14 không nên được xem là duy nhất tối ưu.

### Các ô kết quả còn thiếu

| Hạng mục | Kết quả |
|---|---|
| External dataset | **CHƯA CÓ** |
| Camera thật/production-like holdout | **CHƯA CÓ** |
| Metric theo quả thay vì theo ảnh | **CHƯA CÓ** |
| Blur/tối/lóa/góc chụp/occlusion/background shift | **CHƯA CÓ** |
| Breakdown từng defect trên external | **CHƯA CÓ** |
| Nhiều seed/run, mean±std/CI | **CHƯA CÓ** |
| Calibration ECE/Brier/reliability | **CHƯA CÓ** |
| TFLite INT8 Accuracy/Recall BAD/F1/parity | **CHƯA CÓ** |
| Detector recall/precision và end-to-end fruit decision | **CHƯA CÓ** |

## 7. Export và khả năng triển khai

### Artifact hiện có

| Artifact | Có | Kích thước | Trạng thái xác minh |
|---|---:|---:|---|
| `best_model.keras` | Có | 9.015.163 byte (~8,60 MiB) | Load được; shape/head/params đúng |
| `final_model.keras` | Có | 9.015.163 byte | Chưa so sánh với best |
| `model.tflite` full INT8 | Có | 1.216.992 byte (~1,16 MiB) | Allocate + 1 synthetic invoke thành công |
| ONNX | Không | — | Không cần cho flow TensorFlow→TFLite hiện tại |
| FP16 TFLite | Không | — | CHƯA CÓ |

TFLite được tạo ngày 2026-09-20, sau Keras checkpoint ngày 2026-09-03. Converter mặc đị đọc `training/outputs/models/best_model.keras`, dùng 300 ảnh representative ngẫu nhiên từ toàn bộ `E:/dataset`, seed 42, và ép input/output INT8 (`convert_to_tflite.py:17-41,83-101,203-253`). Tuy nhiên không có conversion manifest/hash, nên timestamp và code mặc đị **không chứng minh tuyệt đối** artifact này sinh từ checkpoint nào.

Chưa có bài test so output Keras–INT8, chưa đánh giá Recall BAD của INT8, và threshold 0,14 của FP32 được copy thẳng sang INT8. Với output quantization bước 1/256, cần đánh giá/hiệu chỉnh threshold trên **validation** sau quantization, sau đó chỉ đánh giá test một lần khi cấu hình đã khóa.

### Raspberry Pi 4

Chưa có bằng chứng chạy trên Pi. Hơn nữa, `app.py:24-30` import `tensorflow` trực tiếp; không hề import/fallback `tflite_runtime`. Điều này mâu thuẫn với README nói chỉ cài `tflite-runtime` và app tự ưu tiên runtime nhẹ. Do đó gói 4 file theo README **sẽ không chạy trong môi trường chỉ có `tflite-runtime`**.

Không có cơ sở để suy ra latency Pi từ con số 3,29 ms/ảnh trong report. Thiếu device, CPU/GPU, threads, batch, warm-up, số lặp, p50/p95, RAM và nhiệt độ.

## 8. Mức độ sẵn sàng cho Early Exit/QRT

- Full Model baseline: **CÓ**, nhưng chưa khóa vì rủi ro split/generalization.
- Multi-Exit: **CHƯA CÓ**.
- Confidence/Entropy baseline: **CHƯA CÓ**.
- Knowledge Distillation: **CHƯA CÓ**.
- EENet: **CHƯA CÓ**.
- QRT-Exit: **CHƯA CÓ**.
- Staged inference bỏ qua phần sau thật sự: **CHƯA CÓ**.
- Full Model vs Multi-Exit FORCE FINAL overhead: **CHƯA CÓ**.

Hướng nghiên cứu Early Exit/QRT hợp lý về mặt logic, nhưng **chưa nên triển khai ngay**. Nếu baseline bị leakage theo quả/session hoặc domain shift camera, exit policy sẽ tối ưu trên nền sai.

Sau khi khóa baseline, có thể probe exit sơ bộ sau các stage/block có downsampling và thay đổi semantic depth của MobileNetV3Small, ví dụ sau các cụm feature map xấp xỉ 56×56, 28×28, 14×14 và 7×7. Mỗi probe dùng head nhẹ GAP→Dense; quyết định vị trí phải dựa trên accuracy/Recall BAD của từng exit, latency thật, overhead FORCE FINAL và calibration. Đây chỉ là vị trí thử, không phải thiết kế đã tối ưu.

## 9. Findings theo mức độ

### P0 — làm sai hoặc vô hiệu kết quả

**Không phát hiện P0 có đủ bằng chứng.** Audit không gán P0 chỉ vì thiếu các công nghệ giai đoạn sau, và không khẳng định leakage khi chưa hoàn thành kiểm tra identity/duplicate.

### P1 — cần xử lý trước khi công nhận baseline

#### P1.1 — Chưa chứng minh split độc lập theo quả/video/session

- **Bằng chứng:** manifest không có group ID; split ngẫu nhiên theo file (`data.py:139-182`).
- **Tác động:** ảnh/frame cùng quả hoặc gần trùng có thể làm internal score lạc quan. Mức 99,91% khiến rủi ro này cần được loại trừ trước.
- **Khắc phục:** bổ sung `fruit_id/session_id/video_id`, exact hash + perceptual near-duplicate audit; group-stratified split; tạo và khóa manifest mới.

#### P1.2 — Chưa có bằng chứng generalization ra camera/domain thật

- **Bằng chứng:** config có external dataset nhưng `outputs/reports/` chỉ có validation và internal_test.
- **Tác động:** không biết baseline có hợp ánh sáng, nền, góc chụp, blur và camera dây chuyền hay không.
- **Khắc phục:** sau khi khóa data/config, đánh giá external và camera holdout theo quả; không dùng holdout để tune.

#### P1.3 — INT8 chưa được xác nhận parity và representative set có nguy cơ dùng test

- **Bằng chứng:** converter quét toàn `E:/dataset`; không có INT8 evaluation report; threshold FP32 được copy (`convert_to_tflite.py:17-41,153-180,223-253`).
- **Tác động:** chưa biết quantization có làm tăng BAD→GOOD; quy trình evaluation không sạch nếu calibration nhìn test distribution.
- **Khắc phục:** representative set chỉ từ train; so probability/prediction FP32–INT8; chọn threshold INT8 trên validation; sau đó chạy test một lần.

#### P1.4 — Chưa thể truy nguyên chắc chắn run → checkpoint → TFLite

- **Bằng chứng:** không có Git; log không có run/phase/config/data hash; không có export manifest; artifact khác ngày nhau.
- **Tác động:** có checkpoint không đồng nghĩa xác nhận checkpoint thuộc đúng config/run đang mô tả.
- **Khắc phục:** version control; mỗi run lưu config, split hash, dependency, seed, commit, checkpoint hash, metric và export hash.

#### P1.5 — Runtime Pi trái với hướng dẫn và chưa được chạy thật

- **Bằng chứng:** `app.py:25` import TensorFlow; README line 217 nói ưu tiên `tflite_runtime`; không có Pi benchmark.
- **Tác động:** cài đặt Pi theo README có thể lỗi import; chưa thể tuyên bố đáp ứng deadline servo.
- **Khắc phục:** adapter runtime LiteRT/TFLite nhẹ, smoke test trên Pi, benchmark protocol đầy đủ.

#### P1.6 — Metric classification không đại diện hiệu năng end-to-end detector→classifier

- **Bằng chứng:** detector HSV/shape có nhiều ngưỡng cứng (`app.py:65-179`), gồm area tối đa 4,5% frame và mean hue 20–43; training lại có `green_orange` là BAD.
- **Tác động:** quả to, quả xanh, quả chạm nhau hoặc đổi ánh sáng có thể không tạo crop; classifier khi đó không có cơ hội quyết định.
- **Khắc phục:** gán nhãn detection/tracking cho video camera đại diện; đo detection recall và decision theo quả/end-to-end trước khi nối servo.

### P2 — cải thiện giai đoạn tiếp theo

#### P2.1 — README đã stale so với code

- **Bằng chứng:** README nói outputs/model chưa có, export FP32 mặc đị, và hỗ trợ `--image`, `--video`, `--save`, `--camera-id`, `--threshold`, FPS; code hiện chỉ có `--camera` và export full INT8.
- **Tác động:** người dùng/người hướng dẫn không thể tái lập lệnh mô tả.
- **Khắc phục:** sinh README từ interface thật hoặc test CLI examples trong CI.

#### P2.2 — Khả năng tái lập và vệ sinh repo yếu

- **Bằng chứng:** không Git, không `.gitignore`, `venv/` nằm trong project, không test/CI, path dataset hard-code.
- **Tác động:** khó review, dễ chuyển nhầm env/artifact, khó lặp lại nghiên cứu.
- **Khắc phục:** khởi tạo Git, ignore env/cache/large artifacts, config path qua CLI, thêm test và run manifest.

#### P2.3 — Threshold tie-break và confidence chưa được đặt trong bài toán chi phí

- **Bằng chứng:** nhiều threshold validation cùng F1/Recall/Accuracy; code không thêm tie-break margin hay cost.
- **Tác động:** 0,14 có thể không bền vững khi quantize/domain shift; probability chưa được calibration.
- **Khắc phục:** xác định cost BAD→GOOD/GOOD→BAD, calibration curve, margin an toàn và threshold protocol cố định.

#### P2.4 — Chưa có robustness/multi-run evidence

- **Bằng chứng:** một training log/checkpoint; không report theo điều kiện ảnh.
- **Tác động:** không biết variance theo seed và failure mode.
- **Khắc phục:** sau khi sửa split, chạy tập seed đã khai báo, report mean/std/CI và slice metrics.

## 10. Đánh giá hướng đi

### **ĐÚNG MỘT PHẦN**

Hướng `Full Model baseline → chọn backbone → Multi-Exit/uncertainty → KD/EENet/QRT → INT8 → staged inference Pi → camera/servo` có trình tự nghiên cứu hợp lý. Project đã làm đúng nhiều phần nền: positive class BAD rõ ràng, class weighting, threshold tách khỏi 0,5 và chọn trên validation, checkpoint trên validation, test có confusion matrix, và đã tạo artifact full INT8.

Tuy nhiên project đang **đi hơi nhanh sang INT8/camera demo khi nền đánh giá chưa khóa**. Trước khi so backbone hay thêm Early Exit, cần chứng minh split group-independent, truy nguyên run, external/camera generalization và FP32↔INT8 parity. Multi-Exit/QRT hiện là roadmap, không phải thiếu sót của giai đoạn này.

## 11. Kế hoạch tiếp theo theo thứ tự ưu tiên

| Bước | Mục tiêu | Việc cần làm | Sản phẩm | Điều kiện hoàn thành |
|---:|---|---|---|---|
| 1 | Chứng minh data split sạch | Bổ sung fruit/session/video ID; exact + perceptual duplicate audit; group-stratified split | Manifest v2 bất biến + leakage report + split hash | 0 group cắt ngang split; duplicate policy rõ; test được niêm phong |
| 2 | Tái lập baseline đáng tin | Git; run manifest; config/env/data/checkpoint hash; chạy lại theo split v2 với các seed đã định | Baseline report mean/std, artifacts truy nguyên được | Run lặp lại được; không dùng test để tune |
| 3 | Kiểm tra generalization | Xây external/camera holdout theo quả; slice blur/tối/lóa/góc/defect | External + slice + fruit-level report | Ngưỡng acceptance do nhóm/người hướng dẫn duyệt |
| 4 | Khóa baseline/backbone | So các backbone trên cùng split/protocol/cost; xem BAD→GOOD và GOOD→BAD | Bảng so sánh công bằng | Chọn Full Model theo quality + compute, không chỉ accuracy |
| 5 | Xác nhận INT8 | Calibration train-only; parity; threshold INT8 trên val; test một lần | INT8 report + export manifest | Recall BAD/F1/FP trong tolerance đã định |
| 6 | Sửa runtime và benchmark Pi | LiteRT/TFLite runtime thật; CLI đúng; protocol warm-up/threads/batch=1/p50/p95/RAM/temp | Pi 4 benchmark report | Chạy ổn định và đạt latency budget servo |
| 7 | Đánh giá detector/camera end-to-end | Gán nhãn video; đo missed fruit/tracking/decision theo quả | End-to-end report | Detection recall và BAD safety đạt target |
| 8 | Bắt đầu Early Exit | Probe exit, confidence/entropy, FORCE FINAL overhead, calibration | Exit trade-off curves | Có exit tiết kiệm latency này mà không phá quality budget |
| 9 | Chỉ thêm KD/EENet/QRT khi có nhu cầu | KD nếu exit yếu; EENet là baseline; QRT dùng uncertainty/quality/multi-frame/deadline | Policy + simulation/replay + ablation | Tốt hơn baseline tĩnh trên quality–latency–deadline |

## 12. Đúng MỘT việc nên làm ngay sau audit

> **Tạo manifest v2 có `fruit_id/session_id/video_id`, audit exact/near-duplicate, rồi chia lại bằng group-stratified split và niêm phong test.**

Đây là việc có giá trị cao nhất vì nó quyết định con số 99,91% có phải baseline đáng tin hay không. Chưa nên thêm Multi-Exit/KD/QRT trước khi hoàn thành việc này.

## 13. Danh sách kiểm tra và giới hạn audit

### Đã chạy

- Liệt kê file/thư mục và tìm `AGENTS.md`: không có `AGENTS.md` trong project.
- Đọc README, requirements, config, toàn bộ 7 file Python của project, log CSV, metric JSON/CSV/text và metadata artifact.
- `git status --short --branch`: thư mục không phải Git repository.
- `python -m py_compile app.py convert_to_tflite.py training/main.py training/orange/*.py`: exit 0.
- `python -m pip check`: `No broken requirements found` trong runtime `venv`.
- `python app.py --help` và `python app.py`: xác nhận CLI chỉ có `--camera`; không mở camera.
- Import version runtime: Python 3.11.9, OpenCV 4.10.0, NumPy 1.26.4, TensorFlow 2.16.1.
- Load `best_model.keras` với `compile=False`: thành công; xác nhận name/params/input/output/head.
- Allocate TFLite interpreter và invoke **một tensor zero synthetic**: thành công; xác nhận dtype/shape/quantization. Thời gian cold single invoke quan sát ~155,8 ms chỉ là smoke test, **không phải benchmark**.
- Đếm manifest/split/class và kiểm tra artifact report hiện có.
- Kiểm tra tên file test/notebook: 0 notebook, 0 test tự động theo convention.

### Không chạy

- Không train/fine-tune lại.
- Không chạy lại internal test hay external test.
- Không đánh giá TFLite trên dataset.
- Không mở camera, video, GUI hay servo.
- Không benchmark Pi 4 hoặc laptop.
- Không kiểm tra near-duplicate/pHash/toàn bộ exact hash; tác vụ hash toàn dataset được dừng khi không còn là kiểm tra nhẹ.
- Không mở nội dung secrets/.env; không có secrets nào được đưa vào báo cáo.

### Giới hạn

- Audit dựa trên source và artifact hiện có; không có Git history hay run registry để xác minh provenance.
- Metric được báo cáo là đọc từ artifact, không được tái tính.
- Sự tồn tại của script/checkpoint không được coi là bằng chứng của một run ngoài những gì log/artifact hỗ trợ.
- Các nhận định về detector, calibration và Pi được gắn nhãn suy luận/chưa có bằng chứng, không trình bày như kết quả thực nghiệm.

