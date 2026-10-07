# Audit chuyên sâu dữ liệu MobileNetV3-Small

**Thời điểm audit:** 2026-09-30  
**Dataset:** `E:/dataset`  
**Manifest:** `training/outputs/splits/dataset_manifest.csv`  
**Phạm vi:** chỉ audit dữ liệu; không sửa ảnh/manifest/source, không chia lại, không train, không inference và không tính lại metric classifier.

## 1. Kết luận ngắn và mức độ chắc chắn

### Quyết định

> **Đã phát hiện các biến thể cùng ảnh nguồn/cùng quả trong khung hình đi xuyên train–validation–test. Cần xây lại split theo group và train lại baseline từ đầu.**

Mức độ chắc chắn của quyết định: **CAO**.

- Không có exact duplicate theo byte hoặc pixel decode.
- Tuy nhiên, review trực quan xác nhận **46 cặp near-duplicate xuyên split**, tạo thành **41 nhóm bằng chứng**, liên quan **87 ảnh duy nhất**.
- Trong số đó có **25 ảnh test** và **26 cặp có test tham gia**. Hiện tượng xuất hiện ở cả 5 lớp gốc, không chỉ một lớp.
- Đây là số tối thiểu được xác nhận từ 60 cặp review có chủ đích; không phải ước lượng đầy đủ tỷ lệ leakage trong 15.000 ảnh.

Kết quả Keras 99,9111% phải được giữ như **kết quả lịch sử của split v1**, nhưng không còn đủ cơ sở để xem là ước lượng độc lập về generalization. Không thể suy ra mức score bị thổi phồng bao nhiêu nếu chưa tạo split group-aware và train lại.

### Trả lời bốn câu hỏi mục tiêu

1. **Dataset đến từ đâu?** **CHƯA CÓ BẰNG CHỨNG.** Không có URL, tên dataset chính thức, tác giả, phiên bản, license, citation, README nguồn hay EXIF. README trong project chỉ mô tả cấu trúc thư mục, không xác nhận nguồn.
2. **Có trùng xuyên split không?** Không có exact duplicate, nhưng **có near-duplicate/cùng ảnh nguồn đã được xác nhận trực quan xuyên split**.
3. **Khôi phục group thật được không?** Chỉ tạo được một số `duplicate_group_id` có bằng chứng. Không đủ thông tin để khôi phục `fruit_id`, `video_id` hoặc `session_id` thật và đầy đủ.
4. **Có cần chia và train lại không?** **CÓ.** Phải hoàn thiện group nguồn/biến thể, tạo split v2 group-aware, sau đó train lại từ đầu. Không được chỉ chuyển ảnh rồi tiếp tục dùng checkpoint cũ.

## 2. Nguồn dữ liệu và quy tắc nhãn

### 2.1 Nguồn, phiên bản và giấy phép

#### ĐÃ XÁC MINH

- `training/config.yaml:8` trỏ tới `E:/dataset`.
- Dataset root chỉ có 5 thư mục lớp; không có README/license/citation/metadata/source document.
- Toàn bộ 15.000 file đều là JPEG 224×224, không file nào chứa JPEG EXIF segment theo phép quét binary.
- File được đặt tên tuần tự theo lớp và phần lớn mtime tập trung trong một đợt ngày 2026-07-29; đây là dấu hiệu dataset đã qua một bước chuẩn bị/batch processing, không phải bằng chứng về nguồn.

#### CHƯA CÓ BẰNG CHỨNG

- URL, tác giả, tổ chức thu thập, camera, ngày chụp, phiên bản, giấy phép sử dụng.
- Dataset công khai nào là nguồn. Dòng `dataset_kaggle_1` trong `training/README_TRAINING.md:176` chỉ là ví dụ cấu hình external dataset, không xác nhận nguồn hiện tại.
- Số ảnh gốc trước xử lý và augmentation recipe. Cấu trúc 3.000 ảnh/lớp, ảnh 224×224 đồng nhất và nhiều biến thể gần nhau là dấu hiệu cần truy nguồn, nhưng audit không tự kết luận toàn bộ 15.000 ảnh là augmentation.

### 2.2 Mapping nhãn

Mapping trong `training/config.yaml:16-20`:

| Lớp gốc | Binary label |
|---|---|
| `fresh_orange` | `GOOD=0` |
| `blackspotted_orange` | `BAD=1` |
| `cankered_orange` | `BAD=1` |
| `green_orange` | `BAD=1` |
| `rotten_orange` | `BAD=1` |

Review mẫu 60 ảnh đại diện, lấy ngẫu nhiên có seed 42 với 6 train + 6 validation mỗi lớp, cho thấy:

- `fresh_orange`: đa số cam màu cam, không thấy bệnh rõ trong mẫu.
- `blackspotted_orange`: có đốm đen/tổn thương bề mặt với mức độ khác nhau.
- `cankered_orange`: thường có vùng sần, sẹo hoặc đổi màu; một số ảnh biểu hiện nhẹ.
- `rotten_orange`: có mốc, thối, biến dạng hoặc hư hỏng rõ.
- `green_orange`: phần lớn thể hiện màu xanh/chưa chín; một số quả đồng thời có vùng tổn thương.

### 2.3 Vấn đề `green_orange → BAD`

**NGHI NGỜ CẦN NGƯỜI PHỤ TRÁCH XÁC NHẬN.**

Nếu mục tiêu thật là “phát hiện khuyết tật/bệnh bề mặt”, màu xanh hoặc chưa chín không tự động chứng minh có khuyết tật. Mapping hiện tại làm `BAD` trở thành khái niệm kết hợp ít nhất hai tiêu chí:

1. bệnh/hư hỏng bề mặt; và
2. độ chín/màu không đạt.

Mapping này có thể hợp lệ nếu mục tiêu sản phẩm là “không đạt tiêu chuẩn thương mại”, trong đó quả xanh bị loại. Nó không đủ bằng chứng nếu báo cáo nghiên cứu định nghĩa BAD chỉ là bệnh bề mặt. Cần văn bản acceptance criteria do người phụ trách xác nhận; audit không tự sửa nhãn.

## 3. Inventory: manifest đối chiếu file thực tế

Nguồn chi tiết: [`dataset_inventory.csv`](dataset_inventory.csv), [`audit_summary.json`](audit_summary.json).

| Kiểm tra | Kết quả |
|---|---:|
| Dòng manifest | 15.000 |
| Path manifest duy nhất | 15.000 |
| File ảnh thực tế dưới dataset root | 15.000 |
| File manifest bị thiếu | 0 |
| Ảnh ngoài manifest | 0 |
| Decode failure | 0 |
| JPEG | 15.000 |
| Độ phân giải 224×224 | 15.000 |
| Có JPEG EXIF segment | 0 |
| Path trùng trong manifest | 0 |
| Tên file trùng trong manifest | 0 |

Phân bố:

| Lớp | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| blackspotted_orange | 2.100 | 450 | 450 | 3.000 |
| cankered_orange | 2.100 | 450 | 450 | 3.000 |
| fresh_orange | 2.100 | 450 | 450 | 3.000 |
| green_orange | 2.100 | 450 | 450 | 3.000 |
| rotten_orange | 2.100 | 450 | 450 | 3.000 |
| **Tổng** | **10.500** | **2.250** | **2.250** | **15.000** |

Manifest v1 được chia bằng `train_test_split` trên từng file, `random_state=42`, stratify theo `original_class` (`training/orange/data.py:161-176`). Nó không có cột group identity.

## 4. Exact duplicate

### Phương pháp

1. **File exact:** SHA-256 toàn bộ byte file.
2. **Decoded-pixel exact:** OpenCV `imdecode(IMREAD_COLOR)`, có áp dụng EXIF orientation mặc định; đổi BGR→RGB; SHA-256 trên prefix `shape|uint8|RGB|` cộng toàn bộ pixel liên tục.

### Kết quả

| Loại | Số nhóm | Ảnh trong nhóm | Ảnh dư nếu giữ 1/group | Nhóm xuyên split | Nhóm mâu thuẫn nhãn |
|---|---:|---:|---:|---:|---:|
| File byte SHA-256 | 0 | 0 | 0 | 0 | 0 |
| Decoded RGB pixel SHA-256 | 0 | 0 | 0 | 0 | 0 |

[`exact_duplicate_groups.csv`](exact_duplicate_groups.csv) có header và 0 dòng dữ liệu, đúng với phạm vi quét đủ 15.000/15.000 ảnh.

**Diễn giải:** dataset không chứa bản sao hoàn toàn giống byte hoặc pixel, nhưng điều này không loại trừ re-encode, thêm nhiễu, đổi sáng hoặc augmentation nhẹ.

## 5. Near-duplicate

### 5.1 Phương pháp

Chi tiết tái lập: [`audit_config.json`](audit_config.json), [`audit_data.py`](audit_data.py).

- pHash 64-bit: RGB→gray, resize 32×32 bằng `INTER_AREA`, OpenCV DCT, lấy block 8×8, so với median của 63 hệ số AC.
- Khoảng cách: Hamming; similarity ghi trong CSV là `1 - distance/64`.
- Ngưỡng ứng viên: Hamming ≤ 6.
- Chỉ tìm cặp xuyên split; loại các cặp đã exact theo decoded pixel.
- Tìm kiếm bằng BK-tree, không dựng ma trận 15.000².
- Giới hạn an toàn 250.000 cặp; thực tế 35.411, **không bị truncate**.

pHash là bộ sinh ứng viên, không phải bằng chứng cùng quả. Nó không hoàn toàn bền vững với crop, rotation, occlusion hoặc biến đổi ảnh mạnh.

### 5.2 Toàn bộ ứng viên

Nguồn: [`near_duplicate_candidates.csv`](near_duplicate_candidates.csv).

| Breakdown | Số cặp |
|---|---:|
| Tổng ứng viên xuyên split | 35.411 |
| train–test | 16.230 |
| train–validation | 15.566 |
| validation–test | 3.615 |
| Hamming 0 | 151 |
| Hamming 2 | 1.555 |
| Hamming 4 | 7.282 |
| Hamming 6 | 26.423 |
| Cùng original class | 23.000 |
| Khác original class | 12.411 |
| Cùng binary label | 28.885 |
| Mâu thuẫn binary label | 6.526 |

Số lượng ứng viên lớn một phần do ảnh đều có bố cục quả tròn ở giữa, nền sáng và cùng kích thước. Review cho thấy nhiều cặp khác nhãn chỉ là false positive do silhouette/nền tương tự; 6.526 cặp mâu thuẫn **không được gọi là label conflict thật** nếu chưa review.

### 5.3 Review trực quan

Đã review 60 cặp bằng contact sheet:

1. 30 cặp đầu gồm cặp gần nhất, cặp mâu thuẫn nhãn và cặp gần ngưỡng.
2. 30 cặp bổ sung phân tầng theo cả 5 lớp, mỗi lớp lấy ứng viên ở khoảng cách 0/2/4 từ vị trí đầu và giữa bucket.

Đây là mẫu có chủ đích để kiểm tra failure mode, không phải mẫu ngẫu nhiên để suy rộng tỷ lệ.

| Trạng thái review | Số cặp |
|---|---:|
| `CONFIRMED_SAME_SOURCE_VARIANT` | 46 |
| `REJECTED_VISUALLY_DIFFERENT` | 11 |
| `REVIEWED_UNCERTAIN` | 3 |
| Chưa review | 35.351 |

46 cặp xác nhận tạo 41 connected evidence groups, gồm 87 ảnh duy nhất:

| Split | Ảnh duy nhất trong nhóm xác nhận |
|---|---:|
| Train | 39 |
| Validation | 23 |
| Test | 25 |

- 26/46 cặp xác nhận có test tham gia.
- Cặp xác nhận theo lớp: fresh 25, blackspotted 5, cankered 5, green 5, rotten 6.
- Các biến thể quan sát được gồm nhiễu/re-encode nhẹ, khác sáng nhỏ, dịch/crop nhẹ hoặc pixel-level augmentation nhưng vẫn cùng quả/bố cục nguồn.
- Các cặp khác nhãn được chọn để review đều bị loại là ảnh khác trong đợt review này; pHash bị đánh lừa bởi bố cục quả tròn giữa nền sáng.

Bằng chứng từng cặp và trạng thái nằm tại [`group_evidence.csv`](group_evidence.csv), [`visual_review_summary.json`](visual_review_summary.json) và [`contact_sheets/`](contact_sheets/).

## 6. Khả năng khôi phục group

### ĐÃ XÁC MINH

- Có thể tạo **41 `duplicate_group_id` tối thiểu, không đầy đủ** từ 46 cặp đã xác nhận.
- Các group này biểu diễn “cùng ảnh nguồn/cùng quả trong framing đã quan sát”, không biểu diễn toàn bộ quan hệ sinh học hoặc phiên thu thập.
- Tên file tuần tự đôi khi đặt các biến thể cạnh nhau, nhưng cũng có biến thể cùng nguồn cách nhau hàng trăm index; không thể dùng khoảng cách tên file làm quy tắc group đáng tin.

### CHƯA CÓ BẰNG CHỨNG

- Không có fruit ID, video/frame ID, session ID, camera ID, timestamp chụp hoặc original-image ID.
- Không thể biết hai góc chụp khác nhau có phải cùng một quả hay không.
- Không thể khôi phục toàn bộ family augmentation bằng một threshold pHash: distance 4 có cả cặp thật lẫn false positive.

### Kết luận group

`duplicate_group_id` có thể xây cho các quan hệ đã xác nhận, nhưng **không tương đương `fruit_id`**. Không thấy duplicate ở một vùng dữ liệu cũng không chứng minh độc lập theo quả/session.

Nếu không tìm được ảnh gốc hoặc augmentation manifest/script, nên:

- dùng dataset hiện tại như development/public-style dataset sau khi group các source variants một cách bảo thủ; và
- thu một camera dataset mới có `fruit_id`, `session_id`, camera/lighting metadata để làm đánh giá theo quả.

## 7. Review nhãn và nguy cơ học đặc trưng phụ

Nguồn mẫu: [`class_sample_selection.csv`](class_sample_selection.csv) và 5 contact sheet `class_sample_*.jpg`.

### Dấu hiệu quan sát được

- Tất cả ảnh đã resize về 224×224 và mất metadata nguồn.
- Có nhiều phong cách chụp/crop: studio nền trắng; nền vải xám/trắng; quả rất nhỏ trên nền xanh nhạt; close-up chiếm gần toàn frame; ảnh có lá; ảnh không lá.
- `fresh_orange` có nhiều ảnh studio màu cam sáng và một family ảnh chụp gần trên nền trắng/xám.
- `green_orange` bị phân biệt mạnh bằng màu xanh, tức màu/độ chín có thể là feature dễ hơn khuyết tật.
- `rotten_orange` có mold/decay rõ, nhưng cũng có một phong cách ảnh quả nhỏ trên nền xanh nhạt xuất hiện nhiều lần.
- Một số nền/phong cách xuất hiện ở nhiều lớp, nên audit không nói background quyết định hoàn toàn nhãn; tuy nhiên phân bố nguồn/crop khác nhau giữa lớp tạo nguy cơ shortcut.

**Không có thí nghiệm attribution/counterfactual**, nên chỉ kết luận có nguy cơ model dùng màu, nền, crop và source style; không khẳng định model đã học shortcut nào.

### HSV detector và `green_orange`

`app.py:74-75` chỉ giữ HSV hue 18–45; `app.py:147` còn yêu cầu mean hue 20–43. Quả xanh bão hòa thường có thể nằm ngoài khoảng này. Vì vậy **suy luận từ code** cho thấy runtime có nguy cơ bỏ qua `green_orange` trước khi classifier nhận crop. Audit không chạy camera và không định lượng recall detector.

## 8. Audit chứng minh gì và chưa chứng minh gì

### ĐÃ XÁC MINH

- Inventory đầy đủ và nhất quán: 15.000 ảnh, không thiếu/thừa, decode được 100%.
- Không có exact duplicate byte/pixel.
- Có source-image variants xuyên split được xác nhận trực quan ở cả 5 lớp.
- Split v1 không độc lập theo ảnh nguồn/augmentation family.
- Test v1 có ít nhất 25 ảnh nằm trong các group xuyên split đã xác nhận từ mẫu review.

### NGHI NGỜ CẦN REVIEW

- Còn nhiều source-image family chưa được group trong 35.351 ứng viên chưa review.
- Mapping green→BAD có thể đúng theo tiêu chuẩn độ chín/thương mại nhưng không được chứng minh là “khuyết tật bề mặt”.
- Dataset có nguy cơ shortcut theo màu, background, crop và source style.

### CHƯA CÓ BẰNG CHỨNG

- Nguồn, giấy phép và augmentation recipe.
- Fruit/video/session identity thật.
- Tổng prevalence leakage trong toàn dataset.
- Mức accuracy 99,91% bị thổi phồng bao nhiêu.
- Chất lượng generalization lên camera thực tế.

## 9. Ảnh hưởng tới kết quả 99,91%

Internal test v1 đã chứa các biến thể cùng ảnh nguồn với train/validation. Điều này vi phạm giả định độc lập cần thiết để dùng test làm ước lượng generalization. Model có thể nhận ra texture, shape, background hoặc framing của source family thay vì chỉ học quy tắc GOOD/BAD tổng quát.

Do audit không chạy classifier và review không bao phủ toàn bộ 35.411 ứng viên, không được phép:

- trừ một con số tùy ý khỏi accuracy;
- tuyên bố toàn bộ test bị leakage; hoặc
- tuyên bố model không có giá trị.

Cách diễn giải đúng:

> **99,9111% là kết quả lịch sử theo ảnh trên split v1 có source-variant leakage đã xác nhận; chưa phải kết quả baseline độc lập theo quả/source group.**

## 10. Có cần chia lại và train lại không?

### Có, với các điều kiện sau

1. Xây group ở mức source image/augmentation family; nếu có metadata gốc, ưu tiên metadata thay vì pHash.
2. Giữ toàn bộ ảnh liên quan trong cùng split.
3. Chia gần 70/15/15 nhưng group non-overlap quan trọng hơn tỷ lệ tuyệt đối.
4. Stratify/kiểm tra phân bố theo cả 5 lớp gốc, không chỉ GOOD/BAD.
5. Niêm phong test v2 và không dùng để chọn checkpoint, threshold hoặc ngưỡng duplicate.
6. Train baseline mới từ đầu. Checkpoint cũ đã học ảnh thuộc train v1 và không hợp lệ để đánh giá trên test v2 nếu các ảnh/group được chuyển vị trí.
7. Giữ split v1, checkpoint và report 99,91% dưới nhãn “historical v1”; không ghi đè.

Nếu không thể khôi phục group đầy đủ, lựa chọn an toàn là dùng dataset này cho development/backbone exploration, còn đánh giá cuối bằng camera dataset mới được thu với fruit/session ID ngay từ đầu.

## 11. Đúng MỘT việc cần làm tiếp theo

> **Tìm lại ảnh gốc và augmentation manifest/script để xây một bảng `source_group_id` đầy đủ cho 15.000 ảnh; nếu không tìm được, lập tức đánh dấu dataset này là development-only và lên kế hoạch thu camera dataset có `fruit_id/session_id`.**

Chưa chia lại ngay bằng pHash thuần túy: review đã chứng minh pHash distance 4 có cả cặp thật và false positive. Group sai cũng có thể làm hỏng split mới.

## 12. Lệnh, dependency, thời gian và artifacts

### Dependency thực tế

- Python 3.11.9
- OpenCV 4.10.0
- NumPy 1.26.4
- Thư viện chuẩn: `csv`, `hashlib`, `json`, `pathlib`, `collections`
- Không cài thêm package; không dùng mạng.

### Lệnh chính

```powershell
.\venv\Scripts\python.exe -m py_compile `
  reports\data_audit\20260930_143802\audit_data.py

.\venv\Scripts\python.exe `
  reports\data_audit\20260930_143802\audit_data.py `
  --manifest training\outputs\splits\dataset_manifest.csv `
  --dataset-root E:\dataset `
  --output-dir reports\data_audit\20260930_143802 `
  --phash-radius 6 `
  --max-near-candidates 250000

.\venv\Scripts\python.exe `
  reports\data_audit\20260930_143802\make_stratified_review.py

.\venv\Scripts\python.exe `
  reports\data_audit\20260930_143802\apply_visual_review.py
```

Automated inventory/hash/pHash scan chạy từ `2026-09-30T14:45:24+07:00` đến `14:48:13+07:00`, tổng 168,87 giây. Sau đó review trực quan 60 cặp và 60 ảnh đại diện lớp bằng các contact sheet.

### Artifacts tạo ra

- [`DATA_AUDIT_REPORT.md`](DATA_AUDIT_REPORT.md)
- [`dataset_inventory.csv`](dataset_inventory.csv) — 15.000 dòng
- [`exact_duplicate_groups.csv`](exact_duplicate_groups.csv) — header, 0 phát hiện
- [`near_duplicate_candidates.csv`](near_duplicate_candidates.csv) — 35.411 cặp và trạng thái review
- [`group_evidence.csv`](group_evidence.csv) — 46 cặp xác nhận, 41 group
- [`visual_review_summary.json`](visual_review_summary.json)
- [`audit_summary.json`](audit_summary.json)
- [`audit_config.json`](audit_config.json)
- [`class_sample_selection.csv`](class_sample_selection.csv)
- [`stratified_visual_review_selection.csv`](stratified_visual_review_selection.csv)
- [`contact_sheets/`](contact_sheets/)
- [`audit_data.py`](audit_data.py)
- [`make_stratified_review.py`](make_stratified_review.py)
- [`apply_visual_review.py`](apply_visual_review.py)

### Giới hạn

- 35.351/35.411 ứng viên chưa được review trực quan; CSV giữ trạng thái `NEEDS_HUMAN_REVIEW`.
- 60 cặp review là mẫu có chủ đích, không dùng để ước lượng tỷ lệ duplicate toàn dataset.
- pHash có false positive và false negative; không phát hiện chắc chắn mọi crop/rotation/strong augmentation.
- Không có domain expert xác minh nhãn bệnh/độ chín.
- Không có metadata nguồn để xác nhận fruit/session identity.
- Không train hoặc đánh giá lại model; kết luận chỉ nói về độ tin cậy của split.

