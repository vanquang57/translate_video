# Video Text Translator

Tự động dịch text cứng (hard-coded subtitle) trong video từ tiếng Trung sang tiếng Việt. Tool phát hiện text trong từng frame, xóa text gốc bằng inpainting, dịch sang tiếng Việt rồi render lại đúng vị trí với style tùy chỉnh, và giữ nguyên audio gốc.

## Yêu cầu hệ thống

- **OS**: Windows 10/11
- **Python**: 3.12 (PaddleOCR hiện chưa hỗ trợ 3.13+)
- **FFmpeg**: bản binary, có trên `PATH`
- **GPU** (tùy chọn): NVIDIA với CUDA 11.8/12.x — không bắt buộc, mọi thứ chạy được CPU
- **RAM**: 16 GB (đủ cho video 1080p ≤ 5 phút)

## Cài đặt

### 1. Cài Python 3.12

Tải installer 64-bit từ <https://www.python.org/downloads/release/python-3127/>. Khi cài tick **Add python.exe to PATH** và **py launcher**.

Kiểm tra:

```cmd
py -3.12 --version
```

### 2. Cài FFmpeg

Cách nhanh nhất qua [scoop](https://scoop.sh):

```cmd
scoop install ffmpeg
```

Hoặc tải bản static từ <https://www.gyan.dev/ffmpeg/builds/> và thêm thư mục `bin\` vào `PATH`. Kiểm tra:

```cmd
ffmpeg -version
ffprobe -version
```

### 3. Clone & cài dependencies

```cmd
git clone <repo-url> "translate video"
cd "translate video"
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Dùng GPU:** uninstall `paddlepaddle` rồi cài bản GPU phù hợp với CUDA của bạn (xem <https://www.paddlepaddle.org.cn/en/install/quick>).
>
> ```cmd
> pip uninstall paddlepaddle
> pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
> ```

### 4. Verify từng module (tùy chọn)

```cmd
python scripts\verify_foundation.py
python scripts\verify_config.py
```

## Cách dùng cơ bản

```cmd
python main.py -i sample.mp4 -o out.mp4
```

CPU mode (mặc định):

```cmd
python main.py -i sample.mp4 -o out.mp4 --compute-mode cpu
```

GPU mode (đã cài `paddlepaddle-gpu`):

```cmd
python main.py -i sample.mp4 -o out.mp4 --compute-mode gpu
```

### Dùng LLM API cho dịch chất lượng cao (9Router / OpenRouter)

Mặc định tool dùng Google Translate (miễn phí, không cần key). Nếu muốn bản dịch chất lượng cao hơn, dùng LLM qua proxy OpenAI-compatible:

#### Cách 1: Dùng 9Router (khuyến nghị — miễn phí, chạy local)

1. Cài 9Router:

   ```cmd
   npm install -g 9router
   9router
   ```

   9Router chạy tại `http://localhost:20128`. Mở dashboard tại <http://localhost:20128> để tạo combo.

2. Tạo combo trong 9Router dashboard (ví dụ đặt tên "free"), thêm các provider bạn muốn (Gemini, OpenAI, Claude, v.v.)

3. Set API key trong file `.env` (9Router không yêu cầu key thật):

   ```dotenv
   LLM_API_KEY=sk-anything
   ```

4. Cấu hình trong `configs/default.yaml`:

   ```yaml
   translator:
     backend: llm
     llm:
       enabled: true
       base_url: "http://localhost:20128/v1"
       model: free              # tên combo trong 9Router
       api_key_env: LLM_API_KEY
       batch_size: 10           # gộp 10 text/request (tiết kiệm request)
       rpm: 30
   ```

5. Chạy:

   ```cmd
   python main.py -i sample.mp4 -o out.mp4 --translator llm
   ```

#### Cách 2: Dùng OpenRouter (trả phí theo usage)

1. Lấy API key tại <https://openrouter.ai/keys>
2. Set trong file `.env`:

   ```dotenv
   LLM_API_KEY=sk-or-v1-...
   ```

3. Cấu hình:

   ```yaml
   translator:
     backend: llm
     llm:
       enabled: true
       base_url: "https://openrouter.ai/api/v1"
       model: google/gemini-2.5-flash-lite
       api_key_env: LLM_API_KEY
       batch_size: 10
       rpm: 30
   ```

#### Fallback tự động

Khi LLM API thất bại (hết quota, timeout, proxy down, v.v.), tool **tự động fallback sang Google Translate miễn phí** — không cần cấu hình gì thêm. Video vẫn được dịch xong, chỉ chất lượng dịch có thể kém hơn ở những câu bị fallback.

#### Batch translation

Cấu hình `batch_size` (mặc định 10) gộp nhiều text vào 1 request API. Ví dụ video có 50 đoạn text → chỉ gọi 5 request thay vì 50. Giảm thời gian chờ và tiết kiệm quota.

```yaml
llm:
  batch_size: 10   # [1, 20] — số text gộp trong 1 lần gọi
```

### Override tham số nhanh

Override một vài tham số nhanh:

```cmd
python main.py -i in.mp4 -o out.mp4 ^
  --ocr-stride 5 --ocr-downscale 2.0 ^
  --inpaint-algo ns --font fonts\BeVietnamPro-Regular.ttf ^
  --font-size-max 80 -v
```

## Cấu hình advanced

Tất cả tham số đặt mặc định trong `configs/default.yaml`. CLI ghi đè giá trị tương ứng. Một vài mục đáng chú ý:

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `compute_mode` | `cpu` hoặc `gpu` | `cpu` |
| `performance.ocr_stride` | OCR mỗi N frame, frame còn lại nội suy | `3` |
| `performance.ocr_downscale` | Resize trước OCR cho nhanh | `1.5` |
| `detector.confidence_threshold` | Bỏ text dưới ngưỡng | `0.5` |
| `tracker.n_inactive` | Số frame trống mới đóng segment | `3` |
| `inpainter.algorithm` | `telea` hoặc `ns` | `telea` |
| `renderer.font_path` | Font tiếng Việt | `fonts/NotoSans-Regular.ttf` |
| `renderer.background_alpha` | Độ mờ background dưới text | `128` |

## Hiệu năng dự kiến

Video 1080p, 3 phút, ocr_stride=3, ocr_downscale=1.5:

| Cấu hình | Thời gian xử lý |
|---|---|
| Ryzen 5 5600 + RTX 3060 Ti (`gpu`) | ≤ 6 phút |
| Intel Ultra 5 134U (`cpu`) | ≤ 30 phút |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'paddleocr'`** — chưa activate venv hoặc chưa `pip install -r requirements.txt`.
- **`ffmpeg not found on PATH`** — cài lại FFmpeg và đảm bảo `ffmpeg`/`ffprobe` nằm trong `PATH`. Mở terminal mới sau khi cài.
- **Lần chạy đầu PaddleOCR tải model rất chậm** — bình thường, model lưu vào `~\.paddleocr\` và sẽ cache cho các lần sau.
- **`base_url phải được cấu hình`** — bạn bật `translator: llm` nhưng chưa set `base_url`. Cần trỏ về proxy (9Router, OpenRouter). Xem hướng dẫn ở trên.
- **LLM API trả 429 / quota exceeded** — hết quota trên provider. Tool sẽ tự fallback sang Google Translate. Nếu muốn tránh, giảm `rpm` hoặc tăng `batch_size`.
- **Dịch bị fallback Google Translate hết** — kiểm tra 9Router có đang chạy không (`http://localhost:20128`), combo có provider nào active không.
- **Text Việt bị tràn khỏi vùng gốc** — giảm `font_size_min` hoặc tăng `box` mask: hiện tại MVP fit vào đúng box gốc; có thể cần style khác (background lớn hơn).
- **Video không có audio output** — nếu input không có audio, output sẽ không có audio (đúng spec). Kiểm tra `ffprobe input.mp4` xem có stream audio không.

## Cấu trúc dự án

```
src/video_text_translator/   # source code
configs/default.yaml         # cấu hình mặc định
fonts/                       # font Việt được bundle sẵn
scripts/                     # script verify từng module
tests/                       # unit + property tests
main.py                      # entry point CLI
requirements.txt
```

## Trạng thái

MVP — chạy end-to-end, không phụ thuộc audio, hỗ trợ CPU và GPU. Các bản nâng cấp tương lai có thể thay OpenCV inpaint bằng ProPainter (chất lượng cao hơn), thay Google Translate bằng NLLB-200 (offline), thêm OCR backend phụ cho text hoạt hình stylized.
