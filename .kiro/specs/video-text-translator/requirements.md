# Requirements Document

## Introduction

Video Text Translator là tool tự động dịch text cứng (hard-coded subtitle/text) trong video từ tiếng Trung sang tiếng Việt. Khác với phụ đề rời (file SRT/ASS), text trong video đã được "burn" vào pixel của frame, có thể xuất hiện ở bất kỳ vị trí nào, với nhiều kiểu font hoạt hình, màu sắc, viền, đổ bóng và hiệu ứng (ví dụ: phóng to theo thời gian). Tool thực hiện end-to-end pipeline: phát hiện text tiếng Trung, theo dõi text qua các frame, xóa text gốc bằng inpainting, dịch sang tiếng Việt, render text Việt với style tùy chỉnh và ghép lại với audio gốc thành video MP4 đầu ra.

Phiên bản này là MVP nhằm chạy được end-to-end với chất lượng chấp nhận được, ưu tiên tốc độ phát triển và khả năng nâng cấp từng module sau. Tool hỗ trợ chạy trên cả CPU và GPU thông qua cấu hình runtime, làm việc với video 1080p độ dài vài phút trở xuống.

## Glossary

- **Video_Text_Translator**: Hệ thống tổng thể thực hiện pipeline dịch text cứng trong video.
- **Hard_Coded_Subtitle**: Text/phụ đề đã được render trực tiếp vào pixel của video, không tách rời thành file phụ đề độc lập.
- **Text_Region**: Vùng hình chữ nhật (bounding box) chứa một đoạn text được phát hiện trong một frame, gồm tọa độ, kích thước và nội dung text được nhận dạng.
- **Text_Segment**: Một đoạn text duy nhất xuất hiện liên tục qua nhiều frame với cùng nội dung (cho phép thay đổi kích thước/vị trí do hiệu ứng), được biểu diễn bởi thời điểm bắt đầu, thời điểm kết thúc và chuỗi các Text_Region theo frame.
- **Detector**: Module phát hiện và nhận dạng text tiếng Trung trong từng frame, sử dụng PaddleOCR (PP-OCRv4).
- **Tracker**: Module liên kết các Text_Region cùng nội dung qua các frame liền kề để tạo thành Text_Segment, dùng kết hợp IoU, center-point và content similarity để chống nhiễu khi text scale.
- **Inpainter**: Module xóa text gốc khỏi frame bằng OpenCV inpainting (TELEA hoặc NS).
- **Translator**: Module dịch nội dung text từ tiếng Trung sang tiếng Việt, mặc định dùng deep-translator với backend Google Translate.
- **Renderer**: Module render text tiếng Việt lên frame bằng Pillow với style preset (background, stroke, shadow) và auto-fit font size theo bounding box gốc.
- **Style_Preset**: Tập hợp các tùy chọn định dạng hiển thị text Việt gồm font family, màu chữ, màu viền, độ dày viền, màu/độ mờ background, màu/độ lệch shadow.
- **Bounding_Box**: Hình chữ nhật bao quanh một Text_Region, biểu diễn bằng (x, y, width, height) tính bằng pixel.
- **Compute_Mode**: Chế độ tính toán cho các module hỗ trợ tăng tốc phần cứng, nhận giá trị "cpu" hoặc "gpu".
- **IoU**: Intersection over Union, tỉ số diện tích giao trên diện tích hợp của hai Bounding_Box, dùng để đo độ trùng lắp.
- **Content_Similarity**: Độ tương đồng giữa hai chuỗi text được nhận dạng, tính bằng normalized edit distance trong khoảng [0, 1].
- **Pipeline_Run**: Một lần chạy end-to-end của Video_Text_Translator từ video đầu vào đến video đầu ra.
- **OCR_Stride**: Số nguyên dương xác định khoảng cách (tính bằng frame) giữa hai frame liên tiếp được đưa vào Detector để OCR. Ví dụ OCR_Stride = 3 nghĩa là chỉ OCR 1 trên mỗi 3 frame, các frame ở giữa được Tracker nội suy từ kết quả OCR gần nhất.
- **OCR_Downscale**: Hệ số thu nhỏ kích thước frame trước khi đưa vào Detector để tăng tốc OCR. Ví dụ OCR_Downscale = 1.5 nghĩa là frame 1080p được resize xuống còn 720p trước OCR, sau đó tọa độ Bounding_Box trả về được scale ngược về kích thước gốc.

## Requirements

### Requirement 1: Đầu vào và đầu ra video

**User Story:** Là người dùng, tôi muốn cung cấp video tiếng Trung và nhận được video MP4 tiếng Việt với audio gốc giữ nguyên, để có thể xem nội dung video bằng tiếng Việt mà không mất phần âm thanh.

#### Acceptance Criteria

1. WHEN người dùng cung cấp đường dẫn video đầu vào hợp lệ và đường dẫn video đầu ra hợp lệ, THE Video_Text_Translator SHALL chạy toàn bộ Pipeline_Run và ghi video MP4 ra đúng đường dẫn đầu ra với phần mở rộng ".mp4".
2. WHERE video đầu vào có audio track, THE Video_Text_Translator SHALL giữ nguyên audio track gốc trong video đầu ra bằng cách copy stream audio mà không thực hiện re-encode.
3. WHERE video đầu vào không có audio track, THE Video_Text_Translator SHALL tạo video đầu ra không có audio track và hoàn tất Pipeline_Run với trạng thái thành công.
4. THE Video_Text_Translator SHALL giữ nguyên độ phân giải (chiều rộng và chiều cao tính bằng pixel), framerate (frames per second) và thời lượng của video đầu vào trong video đầu ra, với sai lệch thời lượng không vượt quá 1 frame và sai lệch framerate không vượt quá 0.01 fps.
5. IF đường dẫn video đầu vào không tồn tại, không phải là file thông thường, hoặc không có quyền đọc, THEN THE Video_Text_Translator SHALL dừng Pipeline_Run trước khi xử lý frame và trả về thông báo lỗi chỉ rõ đường dẫn không hợp lệ và lý do (không tồn tại / không đọc được).
6. IF định dạng video đầu vào không được OpenCV hỗ trợ giải mã hoặc file video bị hỏng (rỗng 0 byte hoặc không mở được), THEN THE Video_Text_Translator SHALL dừng Pipeline_Run và trả về thông báo lỗi nêu rõ định dạng không hỗ trợ hoặc file bị hỏng, đồng thời không tạo file đầu ra.
7. IF thư mục chứa đường dẫn video đầu ra không tồn tại hoặc không có quyền ghi, THEN THE Video_Text_Translator SHALL dừng Pipeline_Run trước khi xử lý frame và trả về thông báo lỗi chỉ rõ đường dẫn đầu ra không ghi được.
8. IF file tại đường dẫn video đầu ra đã tồn tại, THEN THE Video_Text_Translator SHALL ghi đè file đó và đảm bảo nội dung file gốc không còn sau khi Pipeline_Run hoàn tất thành công.
9. IF thời lượng video đầu vào vượt quá 7200 giây (2 giờ) hoặc kích thước file vượt quá 5 GB, THEN THE Video_Text_Translator SHALL dừng Pipeline_Run và trả về thông báo lỗi chỉ rõ vượt quá giới hạn thời lượng hoặc dung lượng cho phép.

### Requirement 2: Phát hiện và nhận dạng text tiếng Trung

**User Story:** Là người dùng, tôi muốn tool tự phát hiện được text tiếng Trung ở bất kỳ vị trí nào trên video, để không phải chỉ định thủ công vùng phụ đề.

#### Acceptance Criteria

1. WHEN một frame được đưa vào Detector, THE Detector SHALL trả về trong vòng tối đa 5 giây một danh sách Text_Region, trong đó mỗi Text_Region gồm Bounding_Box (toạ độ pixel x, y, width, height nằm trong kích thước frame) và nội dung text được nhận dạng cho mỗi vùng text tiếng Trung phát hiện được.
2. THE Detector SHALL quét toàn bộ diện tích frame (từ pixel (0,0) đến (width-1, height-1)) để phát hiện text mà không giới hạn theo vùng cố định ở phía dưới.
3. THE Detector SHALL chỉ giữ lại các Text_Region có nội dung chứa ít nhất một ký tự thuộc dải Unicode CJK Unified Ideographs (U+4E00–U+9FFF) hoặc CJK Unified Ideographs Extension A (U+3400–U+4DBF).
4. THE Detector SHALL gán cho mỗi Text_Region một độ tin cậy nhận dạng là số thực trong khoảng [0.0, 1.0] do PaddleOCR cung cấp.
5. WHERE người dùng cấu hình ngưỡng độ tin cậy tối thiểu trong khoảng [0.0, 1.0], THE Detector SHALL loại bỏ các Text_Region có độ tin cậy thấp hơn ngưỡng đó; nếu người dùng không cấu hình, THE Detector SHALL áp dụng ngưỡng mặc định 0.5.
6. IF Detector không phát hiện được Text_Region nào trong toàn bộ video sau khi xử lý hết tất cả các frame, THEN THE Video_Text_Translator SHALL ghi một log cảnh báo cho biết không có text tiếng Trung được phát hiện và xuất video đầu ra có cùng độ phân giải, cùng số frame, cùng thứ tự frame và nội dung pixel giống hệt video đầu vào.
7. IF PaddleOCR phát sinh lỗi khi xử lý một frame, THEN THE Detector SHALL ghi log lỗi kèm chỉ số frame, trả về danh sách Text_Region rỗng cho frame đó và tiếp tục xử lý các frame còn lại mà không dừng pipeline.

### Requirement 3: Theo dõi text qua các frame có hiệu ứng scale

**User Story:** Là người dùng, tôi muốn tool nhận biết một đoạn text xuất hiện liên tục là cùng một text dù nó đang phóng to/thu nhỏ, để xác định đúng thời gian xuất hiện và biến mất.

#### Acceptance Criteria

1. WHEN Detector trả kết quả cho các frame liên tiếp, THE Tracker SHALL nhóm các Text_Region thành Text_Segment khi thỏa mãn ít nhất một trong hai điều kiện: (a) IoU giữa hai Bounding_Box ≥ 0.5, hoặc (b) Content_Similarity giữa hai text ≥ 0.7 VÀ khoảng cách center-point của Bounding_Box ≤ 10% chiều dài đường chéo của frame.
2. WHILE một Text_Segment đang được theo dõi, THE Tracker SHALL coi hai Text_Region ở hai frame liền kề là thuộc cùng Text_Segment khi Content_Similarity giữa hai text ≥ 0.7 và center-point lệch không quá 10% chiều dài đường chéo của frame, kể cả khi IoU < 0.5 do scale.
3. THE Tracker SHALL gán cho mỗi Text_Segment một thời điểm bắt đầu (timestamp của frame đầu tiên) và thời điểm kết thúc (timestamp của frame cuối cùng) tính bằng giây với độ chính xác tối thiểu 1/framerate.
4. WHEN một Text_Region không khớp với bất kỳ Text_Segment đang hoạt động nào theo các điều kiện ở criterion 1, THE Tracker SHALL khởi tạo một Text_Segment mới bắt đầu từ frame đó với thời điểm bắt đầu là timestamp của frame hiện tại.
5. WHEN một Text_Segment không có Text_Region nào khớp trong N frame liên tiếp (N có giá trị nguyên trong khoảng 1 đến 30, mặc định 3, được cấu hình trước khi xử lý), THE Tracker SHALL đóng Text_Segment đó với thời điểm kết thúc là timestamp của frame khớp gần nhất.
6. THE Tracker SHALL giữ lại danh sách Bounding_Box theo từng frame của mỗi Text_Segment, mỗi entry gồm chỉ số frame, timestamp và tọa độ Bounding_Box, để các module sau biết vị trí và kích thước thay đổi theo thời gian.
7. WHEN một Text_Segment đã bị đóng, THE Tracker SHALL không khớp Text_Region mới vào Text_Segment đó nữa và phải khởi tạo Text_Segment mới nếu nội dung tương tự xuất hiện trở lại.
8. IF một Text_Region có nội dung rỗng hoặc không tính được Content_Similarity với các Text_Segment đang hoạt động, THEN THE Tracker SHALL bỏ qua Text_Region đó, không khởi tạo Text_Segment mới và ghi nhận sự kiện bỏ qua kèm chỉ số frame để truy vết.
9. WHEN số Text_Segment đang hoạt động đồng thời vượt quá 100, THE Tracker SHALL đóng các Text_Segment cũ nhất theo thứ tự thời điểm bắt đầu cho đến khi số Text_Segment đang hoạt động ≤ 100, với thời điểm kết thúc là timestamp của frame khớp gần nhất.

### Requirement 4: Xóa text gốc bằng inpainting

**User Story:** Là người dùng, tôi muốn text tiếng Trung gốc được xóa sạch khỏi video, để khi chèn text Việt vào không bị chồng lên chữ cũ.

#### Acceptance Criteria

1. WHEN một frame chứa Text_Region thuộc các Text_Segment đã xác định, THE Inpainter SHALL tạo mặt nạ (mask) nhị phân bao phủ toàn bộ các Bounding_Box tương ứng trong frame đó.
2. WHEN mặt nạ đã được tạo cho một frame, THE Inpainter SHALL áp dụng OpenCV inpainting trên frame đó với mặt nạ tương ứng, sử dụng bán kính inpainting cấu hình được trong khoảng từ 1 đến 20 pixel, mặc định 3 pixel.
3. THE Inpainter SHALL nới mỗi Bounding_Box trong mặt nạ thêm một biên (padding) cấu hình được trong khoảng từ 0 đến 20 pixel, mặc định 4 pixel, để xử lý phần viền và shadow của text gốc.
4. WHERE người dùng cấu hình thuật toán inpainting với giá trị "telea" hoặc "ns", THE Inpainter SHALL sử dụng đúng thuật toán đó cho toàn bộ quá trình inpainting.
5. WHERE người dùng không cấu hình thuật toán inpainting, THE Inpainter SHALL sử dụng thuật toán mặc định "telea".
6. IF người dùng cấu hình thuật toán inpainting với giá trị khác "telea" và "ns", THEN THE Inpainter SHALL từ chối cấu hình với thông báo lỗi cho biết giá trị không hợp lệ và dừng xử lý trước khi bắt đầu inpainting, giữ nguyên video gốc không thay đổi.
7. THE Inpainter SHALL xử lý mọi Text_Region thuộc các Text_Segment đã xác định cho từng frame trong khoảng thời gian xuất hiện của Text_Segment mà không bỏ sót frame nào.
8. IF một Text_Region nằm vượt ra ngoài biên frame, THEN THE Inpainter SHALL cắt mặt nạ theo biên frame và vẫn áp dụng inpainting cho phần Bounding_Box nằm trong frame.

### Requirement 5: Dịch text từ tiếng Trung sang tiếng Việt

**User Story:** Là người dùng, tôi muốn nội dung text tiếng Trung được dịch tự động sang tiếng Việt, để tôi hiểu được nội dung mà không cần biết tiếng Trung.

#### Acceptance Criteria

1. WHEN một Text_Segment được hoàn thiện và có chuỗi text tiếng Trung với độ dài từ 1 đến 5000 ký tự (sau khi loại bỏ khoảng trắng đầu/cuối), THE Translator SHALL gửi chuỗi text đó tới backend dịch và trả về chuỗi text tiếng Việt tương ứng trong vòng tối đa 10 giây cho mỗi lần gọi backend.
2. THE Translator SHALL sử dụng deep-translator với backend Google Translate làm cấu hình mặc định, với ngôn ngữ nguồn là tiếng Trung và ngôn ngữ đích là tiếng Việt.
3. WHEN nhận được một Text_Segment có chuỗi text trùng khớp chính xác (so sánh phân biệt chữ hoa/thường, sau khi loại bỏ khoảng trắng đầu/cuối) với một Text_Segment đã được dịch trước đó trong cùng một Pipeline_Run, THE Translator SHALL trả về kết quả dịch từ cache nội bộ và SHALL NOT gọi lại backend dịch.
4. IF backend dịch trả về lỗi mạng, lỗi timeout (vượt quá 10 giây), hoặc lỗi quota, THEN THE Translator SHALL thử lại tối đa 3 lần với backoff theo cấp số nhân (lần 1: chờ 1 giây, lần 2: chờ 2 giây, lần 3: chờ 4 giây) trước khi báo lỗi.
5. IF sau 3 lần thử lại Translator vẫn không nhận được bản dịch hợp lệ, THEN THE Translator SHALL trả về chuỗi text tiếng Trung gốc làm kết quả, ghi log một bản ghi cho Text_Segment đó với mức ERROR bao gồm định danh Text_Segment và lý do thất bại, và đánh dấu trạng thái dịch của Text_Segment là "untranslated".
6. IF chuỗi text của Text_Segment là chuỗi rỗng hoặc chỉ chứa các ký tự khoảng trắng (space, tab, newline), THEN THE Translator SHALL trả về chính chuỗi đó không thay đổi và SHALL NOT gọi backend dịch.
7. IF chuỗi text của Text_Segment vượt quá 5000 ký tự, THEN THE Translator SHALL trả về chuỗi text gốc, đánh dấu trạng thái dịch của Text_Segment là "untranslated", và ghi log mức WARNING chỉ ra rằng chuỗi vượt quá độ dài tối đa cho phép.

### Requirement 6: Render text tiếng Việt với style tùy chỉnh

**User Story:** Là người dùng, tôi muốn text tiếng Việt được hiển thị rõ ràng với style do tôi cấu hình (background, viền, đổ bóng), để dễ đọc và phù hợp với tổng thể video.

#### Acceptance Criteria

1. WHEN một Text_Segment đã được dịch sang tiếng Việt và đã có Style_Preset được cấu hình, THE Renderer SHALL render text tiếng Việt lên tất cả các frame nằm trong khoảng thời gian [start_time, end_time] của Text_Segment bằng Pillow theo Style_Preset đó.
2. THE Renderer SHALL hỗ trợ Style_Preset gồm các tùy chọn với phạm vi giá trị: font family (đường dẫn file font hợp lệ tới file .ttf hoặc .otf tồn tại trên hệ thống), kích thước font tối đa (số nguyên trong [8, 512] pixel), kích thước font tối thiểu (số nguyên trong [6, kích thước font tối đa] pixel), màu chữ (RGB, mỗi kênh trong [0, 255]), màu viền (RGB, mỗi kênh trong [0, 255]) và độ dày viền (số nguyên trong [0, 20] pixel), màu background (RGB, mỗi kênh trong [0, 255]) và độ mờ alpha (số nguyên trong [0, 255]), màu shadow (RGB, mỗi kênh trong [0, 255]) và offset (dx, dy) (mỗi giá trị là số nguyên trong [-50, 50] pixel).
3. WHEN render một Text_Segment tại một frame, THE Renderer SHALL auto-fit kích thước font trong khoảng [kích thước font tối thiểu, kích thước font tối đa] sao cho hộp bao của text tiếng Việt (bao gồm cả viền và shadow offset) nằm hoàn toàn trong Bounding_Box của Text_Segment tại frame đó, không vượt quá biên trên cả chiều ngang và chiều dọc, với khoảng cách từ mỗi cạnh text tới cạnh tương ứng của Bounding_Box không nhỏ hơn 0 pixel.
4. WHILE Bounding_Box của một Text_Segment thay đổi kích thước qua các frame do hiệu ứng scale, THE Renderer SHALL tính lại kích thước font cho từng frame theo quy tắc auto-fit ở tiêu chí 3 dựa trên kích thước Bounding_Box hiện tại của frame đó.
5. WHEN render text Việt vào một Bounding_Box, THE Renderer SHALL đặt tâm hộp bao của text trùng tâm Bounding_Box của Text_Segment tại frame đó, với sai lệch không quá 2 pixel theo cả trục ngang và trục dọc.
6. IF Renderer không thể đặt text Việt trong sai lệch 2 pixel theo tâm Bounding_Box do giới hạn của font hoặc layout, THEN THE Renderer SHALL bỏ qua việc render text Việt cho frame đó và ghi log cảnh báo nêu rõ định danh Text_Segment và chỉ số frame bị bỏ qua.
7. IF tại kích thước font tối thiểu mà hộp bao của text Việt vẫn vượt quá Bounding_Box theo chiều ngang hoặc chiều dọc, THEN THE Renderer SHALL bỏ qua việc render text Việt cho frame đó và ghi log cảnh báo nêu rõ định danh Text_Segment, chỉ số frame, kích thước Bounding_Box và kích thước hộp bao text tại font tối thiểu.
8. THE Renderer SHALL chỉ hiển thị text Việt trong khoảng thời gian từ start_time đến end_time của Text_Segment, không hiển thị tại bất kỳ frame nào nằm ngoài khoảng này.
9. WHERE font tiếng Việt được cấu hình không hỗ trợ một ký tự cụ thể trong text Việt, THE Renderer SHALL ghi log cảnh báo nêu rõ ký tự không hỗ trợ và định danh Text_Segment, đồng thời tiếp tục render bằng glyph mặc định của font cho ký tự đó.

### Requirement 7: Cấu hình chạy CPU hoặc GPU

**User Story:** Là người dùng, tôi muốn chuyển đổi giữa chạy CPU và GPU bằng một cấu hình duy nhất, để dùng được tool trên cả máy không có GPU và máy có GPU.

#### Acceptance Criteria

1. THE Video_Text_Translator SHALL nhận cấu hình Compute_Mode với hai giá trị hợp lệ duy nhất là chuỗi "cpu" và "gpu" (so sánh không phân biệt hoa-thường), và SHALL sử dụng giá trị mặc định "cpu" khi Compute_Mode không được cung cấp.
2. WHEN Compute_Mode được đặt là "cpu", THE Detector SHALL khởi tạo PaddleOCR với cờ sử dụng GPU bị tắt trước khi xử lý frame đầu tiên.
3. WHEN Compute_Mode được đặt là "gpu" và môi trường có GPU khả dụng cho PaddleOCR, THE Detector SHALL khởi tạo PaddleOCR với cờ sử dụng GPU bật trước khi xử lý frame đầu tiên.
4. IF Compute_Mode là "gpu" nhưng môi trường chạy không có GPU khả dụng cho PaddleOCR, THEN THE Video_Text_Translator SHALL ghi log cảnh báo nêu rõ lý do không dùng được GPU, tự động fallback Compute_Mode về "cpu", và tiếp tục khởi tạo Detector ở chế độ CPU mà không dừng pipeline.
5. IF việc phát hiện GPU hoặc fallback về CPU thất bại trong giai đoạn khởi tạo, THEN THE Video_Text_Translator SHALL dừng pipeline trước khi xử lý frame, không tạo file đầu ra một phần, và trả về thông báo lỗi nêu rõ nguyên nhân không khởi tạo được Compute_Mode.
6. IF Compute_Mode nhận giá trị khác "cpu" và "gpu" (sau khi chuẩn hoá hoa-thường), THEN THE Video_Text_Translator SHALL dừng pipeline trước khi tải video, không tạo file đầu ra, và trả về thông báo lỗi nêu rõ giá trị Compute_Mode không hợp lệ cùng danh sách các giá trị được chấp nhận.
7. WHERE môi trường chạy chỉ có CPU và Compute_Mode là "cpu", THE Video_Text_Translator SHALL chạy hoàn tất toàn bộ pipeline với Python 3.12 mà không yêu cầu thư viện hoặc driver GPU nào được cài đặt.

### Requirement 8: Pipeline tự động end-to-end

**User Story:** Là người dùng, tôi muốn chạy một lệnh duy nhất là tool tự thực hiện toàn bộ các bước, để không phải can thiệp thủ công giữa các giai đoạn.

#### Acceptance Criteria

1. WHEN người dùng khởi chạy Pipeline_Run với video đầu vào, video đầu ra và file cấu hình hợp lệ, THE Video_Text_Translator SHALL thực hiện tuần tự các bước detection, tracking, inpainting, translation, rendering và đóng gói video theo đúng thứ tự đó mà không yêu cầu thao tác thủ công giữa các giai đoạn.
2. IF video đầu vào không tồn tại, không đọc được, hoặc file cấu hình không hợp lệ, THEN THE Video_Text_Translator SHALL dừng pipeline trước khi bắt đầu giai đoạn detection, ghi log lỗi nêu rõ tham số bị thiếu hoặc không hợp lệ, và trả về mã thoát khác 0.
3. WHILE pipeline đang xử lý, THE Video_Text_Translator SHALL hiển thị thanh tiến độ qua tqdm với thông tin số frame đã xử lý, tổng số frame và tên giai đoạn hiện tại, cập nhật ít nhất sau mỗi frame được xử lý xong.
4. THE Video_Text_Translator SHALL ghi log các sự kiện chính gồm thời điểm bắt đầu và kết thúc của mỗi giai đoạn, tổng số Text_Segment phát hiện, tổng số Text_Segment dịch thành công, tổng số Text_Segment dịch thất bại và đường dẫn tuyệt đối của file video đầu ra.
5. WHEN Pipeline_Run hoàn tất tất cả các giai đoạn và file video đầu ra tồn tại tại đường dẫn đã cấu hình với kích thước lớn hơn 0 byte, THE Video_Text_Translator SHALL trả về mã thoát 0 và in ra stdout đường dẫn tuyệt đối của file video đầu ra.
6. IF tất cả các giai đoạn xử lý hoàn tất nhưng file video đầu ra không tồn tại tại đường dẫn đã cấu hình hoặc có kích thước 0 byte, THEN THE Video_Text_Translator SHALL trả về mã thoát khác 0 và ghi log lỗi nêu rõ file đầu ra bị thiếu hoặc rỗng kèm đường dẫn đã kỳ vọng.
7. IF bất kỳ giai đoạn nào trong pipeline ném ra exception không xử lý được, THEN THE Video_Text_Translator SHALL ghi log stack trace đầy đủ kèm tên giai đoạn xảy ra lỗi, dừng các giai đoạn còn lại và trả về mã thoát khác 0.

### Requirement 9: Hiệu năng và giới hạn vận hành

**User Story:** Là người dùng, tôi muốn tool xử lý ổn định trong giới hạn phần cứng đã thống nhất, để không bị treo hoặc tràn bộ nhớ trên máy của tôi.

#### Acceptance Criteria

1. WHEN video đầu vào có độ phân giải 1920x1080, thời lượng từ 1 giây đến tối đa 5 phút và tốc độ khung hình từ 24 đến 60 fps, THE Video_Text_Translator SHALL hoàn tất Pipeline_Run trên cấu hình AMD Ryzen 5 5600 + RTX 3060 Ti + 16GB RAM với Compute_Mode "gpu" sao cho mức sử dụng RAM hệ thống của tiến trình không vượt quá 12GB và mức sử dụng VRAM không vượt quá 7GB trong suốt quá trình chạy.
2. WHEN video đầu vào có độ phân giải 1920x1080, thời lượng từ 1 giây đến tối đa 5 phút và tốc độ khung hình từ 24 đến 60 fps, THE Video_Text_Translator SHALL hoàn tất Pipeline_Run trên cấu hình Intel Core Ultra 5 134U + 16GB RAM với Compute_Mode "cpu" sao cho mức sử dụng RAM hệ thống của tiến trình không vượt quá 12GB trong suốt quá trình chạy.
3. THE Video_Text_Translator SHALL xử lý frame theo streaming, tại bất kỳ thời điểm nào chỉ giữ trong RAM tối đa số lượng frame bằng kích thước batch hiện hành (từ 1 đến 32 frame) cộng với một bộ đệm đọc/ghi không quá 8 frame, thay vì nạp toàn bộ frame của video vào RAM cùng lúc.
4. WHERE người dùng cấu hình kích thước batch cho Detector trong khoảng từ 1 đến 32 frame, THE Detector SHALL xử lý OCR theo đúng kích thước batch đó cho mọi nhóm frame, ngoại trừ nhóm cuối cùng có thể nhỏ hơn nếu số frame còn lại không đủ một batch đầy.
5. WHERE người dùng tắt chế độ batch hoặc đặt kích thước batch bằng 1, THE Detector SHALL xử lý OCR theo từng frame đơn lẻ và Pipeline_Run vẫn hoàn tất thành công với cùng kết quả định dạng đầu ra như khi chạy ở chế độ batch.
6. IF mức sử dụng RAM hoặc VRAM của tiến trình vượt ngưỡng quy định tại tiêu chí 1 hoặc 2, THEN THE Video_Text_Translator SHALL dừng Pipeline_Run, giải phóng tài nguyên đã cấp phát và trả về thông báo lỗi cho biết đã vượt ngưỡng bộ nhớ kèm cấu hình phần cứng đang dùng.
7. IF người dùng cấu hình kích thước batch nằm ngoài khoảng từ 1 đến 32, THEN THE Video_Text_Translator SHALL từ chối khởi chạy Pipeline_Run và trả về thông báo lỗi cho biết kích thước batch không hợp lệ kèm khoảng giá trị được phép.


### Requirement 10: Tăng tốc OCR bằng frame skipping và downscale

**User Story:** Là người dùng, tôi muốn tool xử lý nhanh hơn đáng kể (đặc biệt trên máy chỉ có CPU), để có thể chạy được video vài phút trong thời gian hợp lý mà vẫn giữ được độ chính xác chấp nhận được.

#### Acceptance Criteria

1. THE Video_Text_Translator SHALL nhận cấu hình OCR_Stride là số nguyên trong khoảng từ 1 đến 10, mặc định 3, xác định khoảng cách frame giữa các lần gọi Detector.
2. WHEN OCR_Stride = 1, THE Detector SHALL được gọi cho mọi frame của video; WHEN OCR_Stride = N với N > 1, THE Detector SHALL chỉ được gọi cho các frame có chỉ số là bội của N (frame 0, N, 2N, ...) và frame cuối cùng của video.
3. WHEN OCR_Stride > 1, THE Tracker SHALL nội suy Bounding_Box và nội dung text cho các frame nằm giữa hai lần OCR liên tiếp dựa trên Text_Segment đang hoạt động, sử dụng Bounding_Box gần nhất theo trục thời gian, và đảm bảo Text_Segment vẫn có entry liên tục cho mọi frame trong khoảng [start_time, end_time].
4. THE Video_Text_Translator SHALL nhận cấu hình OCR_Downscale là số thực trong khoảng từ 1.0 đến 4.0, mặc định 1.5, xác định hệ số thu nhỏ kích thước frame trước khi đưa vào Detector.
5. WHEN OCR_Downscale > 1.0, THE Detector SHALL resize frame xuống kích thước (width / OCR_Downscale, height / OCR_Downscale) bằng phép nội suy bilinear trước khi gọi PaddleOCR, sau đó scale tọa độ Bounding_Box trả về theo đúng hệ số OCR_Downscale để đưa về tọa độ trên frame gốc.
6. WHEN OCR_Downscale = 1.0, THE Detector SHALL không resize frame và xử lý OCR trực tiếp trên kích thước gốc.
7. IF OCR_Stride hoặc OCR_Downscale nhận giá trị nằm ngoài khoảng cho phép, THEN THE Video_Text_Translator SHALL từ chối khởi chạy Pipeline_Run trước khi xử lý frame và trả về thông báo lỗi nêu rõ tên tham số sai cùng khoảng giá trị được phép.
8. WHEN OCR_Stride = 3 và OCR_Downscale = 1.5 trên video 1920x1080 @ 30fps thời lượng tối đa 5 phút, THE Video_Text_Translator SHALL hoàn tất Pipeline_Run trên cấu hình AMD Ryzen 5 5600 + RTX 3060 Ti với Compute_Mode "gpu" trong vòng 6 phút và trên cấu hình Intel Core Ultra 5 134U với Compute_Mode "cpu" trong vòng 30 phút.
9. WHEN OCR_Stride > 1, THE Tracker SHALL điều chỉnh giá trị N của tiêu chí 3.5 (số frame liên tiếp không khớp để đóng Text_Segment) thành max(3, ceil(N_configured * OCR_Stride)) để tránh đóng Text_Segment quá sớm do bỏ qua frame.
