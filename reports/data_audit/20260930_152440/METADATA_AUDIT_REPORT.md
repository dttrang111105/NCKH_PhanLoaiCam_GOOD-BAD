# Audit metadata và nguồn gốc dataset tự chụp

**Thời điểm audit:** 2026-09-30  
**Phạm vi cuối cùng theo chỉ dẫn của chủ dataset:** chỉ `E:/dataset` và manifest/source liên quan trong `E:/orange_project`. Không dùng thư mục khác trên ổ E làm bằng chứng.  
**Chế độ:** chỉ đọc; không sửa ảnh, ADS, timestamp, tên file, manifest hoặc source; không train/inference.

## 1. Kết luận ngắn

### ĐÃ XÁC MINH

1. Toàn bộ 15.000 ảnh hiện tại là JPEG 224×224 với cấu trúc marker đồng nhất và chỉ có metadata JPEG cơ bản `JFIF 1.01`.
2. Không ảnh nào còn EXIF, IPTC, XMP, extended XMP, ICC profile, Photoshop APP13, Adobe APP14 hoặc JPEG comment.
3. Không còn `DateTimeOriginal`, Make, Model, lens, orientation, software/creator tool, GPS hoặc capture time trong payload JPEG.
4. Cả 15.000 file đều có NTFS alternate data stream `Zone.Identifier`:
   - `ZoneId=3`;
   - `ReferrerUrl=C:\Users\ADMIN\Downloads\OrangeClassificationDataset.zip`;
   - không có `HostUrl`.
5. Tên file ở cả 5 lớp đều có dạng `<original_class>_0000.jpg` đến `<original_class>_2999.jpg`, liên tục, không thiếu index.
6. File system time tạo thành hai dải batch rất hẹp và tuần tự theo lớp; chúng không phải bằng chứng thời gian chụp.
7. Manifest hiện tại chỉ lưu `image_path,file_name,original_class,label,binary_label,split`; không có capture/source identity.

### SUY LUẬN CÓ GIỚI HẠN

- Ảnh hiện tại đã đi qua một quy trình chuẩn hóa/export đồng nhất trước hoặc trong lúc đóng gói archive. Bằng chứng là toàn bộ ảnh cùng 224×224, cùng marker layout, cùng JFIF settings, cùng naming pattern và không còn capture metadata.
- `Zone.Identifier` cho thấy file hiện tại được giải nén/chuyển từ archive tên `OrangeClassificationDataset.zip` có Mark of the Web. Nó là dấu vết truyền tải, không phải bằng chứng archive thuộc bên thứ ba.
- Metadata chụp có thể đã bị loại khi resize/convert/export trước lúc tạo ZIP, nhưng không thể xác định bước hoặc công cụ cụ thể từ file hiện tại.

### CHƯA CÓ BẰNG CHỨNG

- Không có metadata máy ảnh hoặc ngày chụp nào còn dùng được trong `E:/dataset`.
- Không có mapping chắc chắn từ file hiện tại về ảnh gốc, video, quả, phiên chụp hoặc camera.
- Không thể xác định `fruit_id`, `session_id`, `capture_time` hoặc `camera_id` từ tên file, index, màu ảnh, độ giống ảnh hay Windows timestamp.
- Không biết archive `OrangeClassificationDataset.zip` còn tồn tại hay bên trong có ảnh gốc/manifest/script hay không; phạm vi hiện tại không kiểm tra thư mục Downloads.

**Kết luận:** metadata chụp không còn trong các JPEG hiện tại. Dấu vết nguồn duy nhất còn dùng được là provenance ở mức **archive truyền tải**, không phải provenance ở mức **lần chụp**.

## 2. Phạm vi và phương pháp

### Phạm vi đã quét

- `E:/dataset/**/*.jpg`: 15.000 file.
- `E:/orange_project/training/outputs/splits/dataset_manifest.csv`.
- Source/config/tài liệu trong `E:/orange_project` để tìm tên archive, script capture/resize/rename và các trường ID.
- Báo cáo audit trước:
  - `reports/AUDIT_MOBILENETV3_SMALL_CURRENT_STATE.md`;
  - `reports/data_audit/20260930_143802/DATA_AUDIT_REPORT.md`.

Theo yêu cầu bổ sung của chủ dataset, không sử dụng các thư mục khác trên ổ E trong kết luận này.

### Phương pháp JPEG

Script [`audit_jpeg_metadata.py`](audit_jpeg_metadata.py) đọc marker JPEG mà không decode/rewrite pixel:

- APP0/JFIF;
- APP1/EXIF;
- APP1/XMP và extended XMP;
- APP2/ICC;
- APP13/Photoshop/IPTC;
- APP14/Adobe;
- COM comment;
- SOF width/height/components/precision;
- các EXIF tag thông dụng: Make, Model, Software, Orientation, DateTime, DateTimeOriginal, DateTimeDigitized, offset time, lens và GPS presence.

Script cũng đọc NTFS ADS `Zone.Identifier` ngay trên mỗi file và đối chiếu manifest.

## 3. Kết quả metadata JPEG

Nguồn chi tiết: [`metadata_inventory.csv`](metadata_inventory.csv), [`metadata_summary.json`](metadata_summary.json).

| Trường/segment | File có dữ liệu | Ý nghĩa |
|---|---:|---|
| JPEG hợp lệ | 15.000 | Tất cả file parse được |
| Kích thước 224×224 | 15.000 | Kích thước pixel hiện tại |
| JFIF 1.01 | 15.000 | Container metadata cơ bản |
| JFIF density | 15.000 | Unit=0, X=1, Y=1; không có DPI vật lý |
| EXIF | 0 | Không có metadata camera/capture EXIF |
| IPTC | 0 | Không keyword/byline/date IPTC |
| XMP | 0 | Không creator tool/create date/camera XMP |
| Extended XMP | 0 | Không có |
| ICC profile | 0 | Không có profile màu nhúng |
| Photoshop APP13 | 0 | Không có |
| Adobe APP14 | 0 | Không có |
| JPEG COM comment | 0 | Không có |
| Camera Make | 0 | Không xác định |
| Camera Model | 0 | Không xác định |
| DateTimeOriginal/capture time | 0 | Không xác định |
| Orientation | 0 | Không có tag orientation |
| Software/CreatorTool | 0 | Không xác định encoder |
| GPS | 0 | Không có |

Tất cả ảnh có cùng marker signature trước vùng entropy:

```text
FFE0-FFDB-FFDB-FFC0-FFC4-FFC4-FFC4-FFC4-FFDA
```

Đây là dấu hiệu pipeline export đồng nhất, nhưng marker layout không nhận diện duy nhất một phần mềm. Không được dùng nó để tuyên bố ảnh được tạo bởi OpenCV, Pillow hay công cụ cụ thể nào.

### Orientation và màu

- Không có EXIF orientation. Pixel đã được lưu ở dạng vuông 224×224; không thể biết ảnh nguồn từng có orientation nào.
- Không có ICC profile. Ứng dụng thường sẽ xử lý chúng như ảnh RGB không quản lý màu/sRGB-like, nhưng file không chứng minh profile nguồn hoặc phép chuyển màu đã dùng.

## 4. NTFS Zone.Identifier và dấu vết archive

### ĐÃ XÁC MINH

Toàn bộ 15.000 file có:

```ini
[ZoneTransfer]
ZoneId=3
ReferrerUrl=C:\Users\ADMIN\Downloads\OrangeClassificationDataset.zip
```

- `ZoneId=3` là dấu Mark of the Web cho Internet Zone trên Windows.
- `ReferrerUrl` trỏ tới archive cục bộ mang tên `OrangeClassificationDataset.zip` trong thư mục Downloads tại thời điểm dấu vết được tạo.
- Không file nào có `HostUrl`, nên ADS không cho biết domain hoặc URL tải xuống.

### Diễn giải đúng

ADS này chứng minh provenance ở mức “các file đã đi qua archive có tên trên”. Nó **không chứng minh**:

- dataset do người khác tạo;
- ảnh không phải tự chụp;
- URL Internet cụ thể;
- archive vẫn còn tồn tại;
- archive có ảnh gốc độ phân giải cao.

Xác nhận của chủ dataset rằng ảnh do tự chụp được ghi nhận là thông tin từ người sở hữu, nhưng không thể kiểm chứng bằng metadata chụp trong JPEG hiện tại. Một archive tự tạo rồi gửi qua cloud/chat/email và tải lại cũng có thể nhận ZoneId=3.

NTFS ADS không nằm trong payload JPEG và có thể mất khi copy sang filesystem không hỗ trợ ADS. Vì vậy nó không phù hợp làm ID lâu dài.

## 5. Tên file, cấu trúc thư mục và timestamp

### 5.1 Tên file

Mỗi lớp có đúng 3.000 tên:

```text
fresh_orange_0000.jpg ... fresh_orange_2999.jpg
blackspotted_orange_0000.jpg ... blackspotted_orange_2999.jpg
cankered_orange_0000.jpg ... cankered_orange_2999.jpg
green_orange_0000.jpg ... green_orange_2999.jpg
rotten_orange_0000.jpg ... rotten_orange_2999.jpg
```

Không có gap hoặc tên lệch pattern. Pattern này chứng minh đã có bước rename/index theo lớp. Nó không chứa separator hay trường nào có căn cứ để giải mã thành fruit/session/camera/time. Audit near-duplicate trước cũng cho thấy biến thể liên quan không phải lúc nào nằm cạnh nhau theo index.

**Không được dùng index làm `fruit_id`.**

### 5.2 Windows CreationTime

| Lớp | Min | Max |
|---|---|---|
| blackspotted | 2025-04-27 06:43:10 +07 | 06:43:14 |
| cankered | 06:43:14 | 06:43:18 |
| fresh | 06:43:18 | 06:43:22 |
| green | 06:43:22 | 06:43:26 |
| rotten | 06:43:26 | 06:43:30 |

15.000 file có CreationTime chỉ trải 20 giây và tuần tự theo thư mục lớp. Đây là timestamp của thao tác file/bulk extraction hoặc timestamp được tool bảo toàn/đặt lại; không thể là capture time đáng tin.

### 5.3 LastWriteTime và directory time

| Lớp | LastWriteTime min | Max |
|---|---|---|
| blackspotted | 2026-07-29 15:41:47 +07 | 15:42:24 |
| cankered | 15:42:24 | 15:42:59 |
| fresh | 15:42:59 | 15:43:34 |
| green | 15:43:34 | 15:44:05 |
| rotten | 15:44:05 | 15:44:39 |

Directory `E:/dataset` có CreationTime `2026-07-29 15:41:47`; các class directory cũng được tạo tuần tự trong cùng dải này. Việc CreationTime của file lại là năm 2025, sớm hơn directory chứa nó, minh họa rõ rằng Windows/file-tool timestamp có thể được phục hồi hoặc gán lại và không phải chronology chụp.

LastWriteTime có thể đến từ resize, convert, archive metadata, extraction, copy hoặc thao tác tool. Không có capture metadata độc lập để đối chiếu, nên không được gọi là ngày chụp.

## 6. Manifest và khả năng liên kết về nguồn

Manifest hiện có:

```text
image_path,file_name,original_class,label,binary_label,split
```

Các trường capture provenance còn thiếu:

- `source_file_name` / `source_relative_path`;
- `source_sha256`;
- `source_group_id` hoặc `augmentation_parent_id`;
- `fruit_id`;
- `session_id`;
- `camera_id`;
- `capture_time` và timezone;
- `frame_id` / `video_id`;
- `view_id` / góc chụp;
- lighting/background/setup ID;
- transform/resize/export recipe;
- operator và label-review status.

### Có liên kết chắc chắn tới nguồn nào không?

- Liên kết tới archive name: **CÓ, mức transfer provenance**, qua Zone.Identifier.
- Liên kết tới file ảnh gốc cụ thể: **KHÔNG**.
- Liên kết tới video/session/quả/camera: **KHÔNG**.
- Liên kết số thứ tự filename tới group: **KHÔNG CÓ CĂN CỨ**.

Trong `E:/orange_project` không có archive, video/raw image, XLSX/notebook, capture log hoặc script resize/rename tạo 15.000 file. `training/orange/data.py:260-266` resize ảnh trong pipeline TensorFlow lúc đọc; augmentation ở `data.py:280-296` chạy trong training pipeline và không ghi ảnh augmentation xuống `E:/dataset`. Vì vậy source hiện tại không giải thích bước tạo các JPEG 224×224 trên disk.

## 7. Metadata có bị mất khi resize/convert không?

### ĐÃ XÁC MINH

Metadata capture không có trong file hiện tại.

### SUY LUẬN

Nhiều thư viện resize/export chỉ ghi pixel và JFIF tối thiểu nếu không chủ động copy EXIF/ICC/XMP. Cấu trúc hiện tại phù hợp với tình huống đó.

### CHƯA CÓ BẰNG CHỨNG

Không thể phân biệt chắc chắn giữa các khả năng:

1. ảnh nguồn vốn không có capture metadata;
2. metadata bị loại khi resize/convert;
3. metadata bị loại khi rename/export trước khi đóng ZIP;
4. một tool tạo lại JPEG từ pixel đã xử lý.

ZIP thông thường lưu byte file và không tự tái mã hóa JPEG. Nếu archive chỉ chứa đúng các file hiện tại thì metadata có khả năng đã mất trước khi hoặc lúc tạo archive; nhưng chưa kiểm tra archive nên đây vẫn là suy luận.

## 8. Metadata nào dùng được cho ID?

| Candidate | Dùng được? | Lý do |
|---|---|---|
| EXIF DateTimeOriginal | Không | Không tồn tại |
| EXIF Make/Model | Không | Không tồn tại |
| EXIF/GPS | Không | Không tồn tại |
| XMP/IPTC | Không | Không tồn tại |
| Windows CreationTime | Không | Batch 20 giây, có thể restore/reset |
| LastWriteTime | Không | Batch xử lý 172 giây, không phải capture evidence |
| Filename index | Không | Chỉ là sequence theo lớp; không có schema group |
| Folder class | Chỉ dùng cho nhãn | Không định danh quả/session |
| JFIF/marker signature | Không | Format/export fingerprint rất yếu |
| Zone.Identifier | Chỉ provenance archive | Cho biết archive name/Internet Zone, không cho capture identity |

Kết luận trực tiếp:

- `fruit_id`: **CHƯA KHÔI PHỤC ĐƯỢC**.
- `session_id`: **CHƯA KHÔI PHỤC ĐƯỢC**.
- `capture_time`: **CHƯA KHÔI PHỤC ĐƯỢC**.
- `camera_id`: **CHƯA KHÔI PHỤC ĐƯỢC**.
- `source_archive_hint`: có thể ghi nhận là `OrangeClassificationDataset.zip`, nhưng phải kèm mức tin cậy “transfer provenance only”.

## 9. Quy trình đề xuất cho ảnh mới

Với mỗi đợt chụp mới:

1. Tạo `session_id` trước khi chụp, ví dụ ID ngẫu nhiên/UUID hoặc mã ngày + sequence do hệ thống cấp; không suy từ mtime.
2. Gắn `fruit_id` vật lý duy nhất cho từng quả trước khi xoay/chụp nhiều góc.
3. Giữ nguyên file gốc, không rename/resize đè lên; tính SHA-256 ngay khi ingest.
4. Tạo manifest append-only gồm:
   - `source_sha256`, source path;
   - `fruit_id`, `session_id`, `camera_id`;
   - capture timestamp có timezone;
   - frame/view ID, góc, khoảng cách;
   - background/lighting/setup;
   - label, labeler, review state.
5. Mỗi ảnh derived 224×224 phải có `derived_id`, `parent_source_sha256`, transform recipe và tool version.
6. Split theo `fruit_id` hoặc session trước khi augmentation; mọi derived image theo parent group.
7. Lưu manifest, source originals và derived dataset bằng version/hash; không dựa vào EXIF làm nguồn duy nhất.

### Dataset cũ

- Giữ nguyên byte và ADS hiện tại làm bằng chứng.
- Không tạo fruit/session ID giả.
- Nếu tìm lại archive/ảnh nguồn, map bằng SHA-256/pixel/perceptual evidence và lưu confidence.
- Nếu không tìm được mapping đầy đủ, dùng dataset cũ ở mức development và dùng bộ chụp mới có ID cho đánh giá theo quả.

## 10. Đúng một việc nên làm tiếp theo

> **Xin phép kiểm tra read-only file `C:/Users/ADMIN/Downloads/OrangeClassificationDataset.zip` được Zone.Identifier chỉ ra, nếu file còn tồn tại, để inventory member names, metadata, ảnh gốc độ phân giải cao và manifest/script bên trong rồi map lại với `E:/dataset`.**

Đây là dấu vết provenance cụ thể duy nhất còn lại. Không nên tiếp tục suy `fruit_id` từ tên file hoặc pHash trước khi kiểm tra archive.

## 11. Lệnh, thời gian và artifacts

### Lệnh chính

```powershell
.\venv\Scripts\python.exe -m py_compile `
  reports\data_audit\20260930_152440\audit_jpeg_metadata.py

.\venv\Scripts\python.exe `
  reports\data_audit\20260930_152440\audit_jpeg_metadata.py `
  --dataset-root E:\dataset `
  --manifest training\outputs\splits\dataset_manifest.csv `
  --output-dir reports\data_audit\20260930_152440

Get-ChildItem E:\dataset -Recurse -File -Filter *.jpg |
  Get-Item -Stream *
```

Quét JPEG + ADS lần cuối hoàn thành 15.000 file trong khoảng 45,07 giây. Không cài package; parser chỉ dùng Python standard library.

### Artifacts mới

- [`METADATA_AUDIT_REPORT.md`](METADATA_AUDIT_REPORT.md)
- [`metadata_inventory.csv`](metadata_inventory.csv) — 15.000 dòng, 54 cột
- [`metadata_summary.json`](metadata_summary.json)
- [`audit_jpeg_metadata.py`](audit_jpeg_metadata.py)

## 12. Giới hạn audit

- Không mở hoặc tìm file trong Downloads theo chỉ dẫn không quét thư mục khác.
- Không kiểm tra filesystem history, shadow copies, cloud history, điện thoại/camera hoặc backup.
- Không thể khôi phục metadata đã bị loại khỏi JPEG chỉ bằng file hiện tại.
- Zone.Identifier có thể bị mất khi copy; kết quả chỉ mô tả trạng thái file tại thời điểm audit.
- User xác nhận dataset tự chụp; audit không có capture metadata độc lập để xác thực hoặc bác bỏ tuyên bố đó.

