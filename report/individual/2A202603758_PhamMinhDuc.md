# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603758
- **Họ và tên:** Phạm Minh Đức
- **SEED** (in ở ô đầu notebook): 7233

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng tập test (SEED = 7233):

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 0.957 | 0.812 | 0.879 | 0.889 | 0.1s | ~18,500 | 0.12 ms | 0.28 ms | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.938 | 0.841 | 0.886 | 0.895 | 1.8s | ~14,200 | 0.18 ms | 0.45 ms | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.962 | 0.855 | 0.905 | 0.900 | 8.2s (encode) | ~85 | 11.20 ms | 18.50 ms | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.966 | 0.913 | **0.939** | **0.950** | 8.3s | ~120 | 8.40 ms | 15.10 ms | $0.000 |
| **3c. Encoder (MiniLM) + MLP head**| 0.955 | 0.913 | 0.933 | 0.950 | 8.7s | ~110 | 8.80 ms | 16.00 ms | $0.000 |
| **4a. LLM nhỏ zero-shot (`gpt-6-luna`)** | 0.950 | 0.950 | 0.950 | 0.950 | 0.0s | ~1.2 | 680 ms | 1,250 ms | $0.016 |
| **4b. LLM lớn zero-shot (`gpt-6.1-terra`)**| 0.952 | 1.000 | 0.975 | 0.975 | 0.0s | ~0.8 | 950 ms | 1,820 ms | $0.160 |
| **4c. LLM nhỏ few-shot (`gpt-6-luna`)** | 0.952 | 1.000 | 0.975 | 0.975 | 0.0s | ~1.0 | 780 ms | 1,420 ms | $0.065 |

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có sự khác biệt hoặc mang lại bài học sâu sắc nhất:

1. **`ml_lr` (Learning Rate với Logistic Regression):**
   - *Dự đoán:* lr=0.001 loss giảm rất chậm; lr=50 loss dao động dữ dội hoặc vọt lên cực lớn.
   - *Thực tế:* Với lr=0.001, loss sau 300 epoch chỉ giảm từ 0.693 xuống ~0.550 (chưa hội tụ). Với lr=50, loss không chỉ dao động mà bão hoà nhanh ở mức cao (~0.45), hàm sigmoid bị đẩy về 0 và 1 tuyệt đối khiến gradient triệt tiêu và không tối ưu thêm được.
   - *Lý do:* Tốc độ học quá lớn gây overshooting qua đáy hàm loss, khiến điểm tham số rơi vào vùng phẳng bão hòa của hàm sigmoid.

2. **`dl_curve` (MLP vs Logistic Regression):**
   - *Dự đoán:* MLP nhiều tầng nơ-ron và 128k tham số sẽ có F1 vượt trội hơn hẳn mô hình tuyến tính LogReg (~0.93 vs ~0.88).
   - *Thực tế:* F1 của MLP chỉ đạt 0.886, gần như tương đương với LogReg (0.879), đồng thời val loss bắt đầu tăng trở lại sau epoch 8 (overfitting).
   - *Lý do:* Đầu vào là túi từ BoW rời rạc và thưa thớt, dữ liệu train nhỏ (~3.600 tin), ranh giới phân loại phần lớn mang tính tuyến tính theo sự hiện diện từ khoá, nên mạng sâu nhiều tham số chỉ học vẹt mà không tìm thêm được quy luật phi tuyến hữu ích nào.

3. **`llm_models` (Model lớn vs Model nhỏ trên SUB 40 tin):**
   - *Dự đoán:* Model lớn (`gpt-6.1-terra`) sẽ bỏ xa model nhỏ (`gpt-6-luna`) về F1 (ví dụ 0.95 vs 0.85).
   - *Thực tế:* Cả hai model đều đạt kết quả rất sát sao trên tập SUB (0.975 vs 0.950, chỉ lệch nhau đúng 1 tin nhắn), trong khi chi phí của model lớn đắt gấp 10 lần.
   - *Lý do:* Tác vụ phân loại nhị phân tin nhắn ngắn không đòi hỏi khả năng suy luận đa tầng phức tạp; model nhỏ đã đủ năng lực hiểu ngữ nghĩa câu SMS thông thường.

---

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất:** **1. BoW + LogReg** và **2. BoW + MLP**. F1 trên tập test thường đạt ~0.88 nhưng khi sang 50 case khó đã tụt dốc nghiêm trọng xuống chỉ còn ~0.52 - 0.58.
- **Nhóm case gây khó khăn nhất:**
  + *Né từ khoá (obfuscation):* Tỷ lệ đoán đúng của BoW chỉ đạt **0% - 16%**.
  + *Tưởng spam nhưng không phải (ham_looks_spam):* BoW bị bẫy False Positive hàng loạt (chỉ đúng ~30%).
- **Hai case cụ thể và cơ chế gây sai:**
  1. **Case #30 (Né từ khoá - nhãn thật: SPAM):**
     `"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"`
     *Cơ chế sai của BoW:* Người gửi tách các chữ cái bằng dấu cách (`F R E E`). Hàm tokenize tách mỗi ký tự thành một token đơn lẻ (`f`, `r`, `e`, `e`). Từ "free" hoàn chỉnh biến mất khỏi biểu diễn BoW, khiến vector $X$ toàn số 0 tại các đặc trưng spam, mô hình đoán nhầm thành ham.
  2. **Case #1 (Tưởng spam nhưng là ham - nhãn thật: HAM):**
     `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"`
     *Cơ chế sai của BoW:* Tin nhắn chứa hàng loạt từ khoá có trọng số dương rất cao trong tập train spam: `congrats`, `won`, `call`. Phép tính $z = w \cdot x + b$ cộng dồn toàn bộ trọng số dương này đẩy xác suất $p \approx 0.98$, mô hình chặn nhầm tin nhắn cá nhân của bạn bè.

---

## 5. Thử thách cá nhân

- **Đề bài theo SEED (SEED = 7233, Đề #4):**
  > *"Embedding · Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?"*
- **Cách làm:**
  Sử dụng ma trận biểu diễn embedding $E_{tr}$, nhãn $y_{tr}$ và tập kiểm tra $E_{te}$. Lần lượt gọi hàm `knn_predict(E_tr, y_tr, E_te, k=k)` với các giá trị $k \in \{1, 5, 25\}$, sau đó tính toán precision, recall và F1 qua hàm `metrics`.
- **Kết quả thu được:**
  + $k = 1$: Precision = 0.941, Recall = 0.855 $\rightarrow$ **F1 = 0.896**
  + $k = 5$: Precision = 0.962, Recall = 0.855 $\rightarrow$ **F1 = 0.905** (kết quả tốt nhất)
  + $k = 25$: Precision = 0.970, Recall = 0.768 $\rightarrow$ **F1 = 0.857** (Recall sụt giảm mạnh)
- **Giải thích cơ chế:**
  + *Khi $k$ quá nhỏ ($k=1$):* Mô hình có phương sai cao (High Variance). Ranh giới quyết định bị gồ ghề và cực kỳ nhạy cảm với các điểm ngoại lai (outliers) hoặc các mẫu bị gán nhãn sai trong tập huấn luyện.
  + *Khi $k$ quá lớn ($k=25$):* Mô hình có độ chệch cao (High Bias). Do tập dữ liệu mất cân bằng nghiêm trọng (~87% ham), khi lấy tới 25 láng giềng gần nhất, số lượng tin ham sẽ dễ dàng áp đảo số lượng tin spam trong hầu hết các vùng không gian, khiến các mẫu spam biên giới bị bỏ sót (Recall tụt từ 85.5% xuống 76.8%).

---

## 6. Phương án cho sản phẩm thật

- **Yêu cầu bài toán:** Lọc 10 triệu SMS/ngày (~116 tin/giây trung bình, giờ cao điểm có thể lên tới 500 tin/giây), yêu cầu nghiêm ngặt **SLA độ trễ p95 < 50 ms/tin**.
- **Phương án lựa chọn:** **3b. Encoder (MiniLM) + LogReg** (hoặc kiến trúc phân tầng Router).
- **Dẫn chứng số liệu:**
  + *Độ trễ:* p95 của Encoder + LogReg là **15.10 ms/tin**, hoàn toàn thỏa mãn SLA < 50 ms. Ngược lại, LLM có p95 từ 1.250 ms đến 1.820 ms (chậm gấp gần 100 lần SLA cho phép).
  + *Chất lượng:* F1 test đạt **0.939**, cao nhất trong số các phương án xử lý nội bộ, có khả năng khái quát hoá vượt trội trước các biến thể từ khoá.
  + *Chi phí API:* **$0.00 USD**, toàn bộ mô hình MiniLM (~90MB) có thể chạy trực tiếp trên CPU/GPU của hạ tầng máy chủ nhà mạng mà không tốn phí bản quyền API bên thứ ba.
- **Đánh đổi chấp nhận:**
  + Chấp nhận độ trễ suy luận cao hơn BoW (15 ms so với 0.3 ms của BoW), nhưng 15 ms vẫn nằm rất sâu trong ngưỡng an toàn 50 ms của SLA.
  + Nếu muốn bảo vệ tuyệt đối trước các đòn tấn công *Prompt Injection* tinh vi, ta kết hợp thêm bộ lọc tiền xử lý dựa trên quy tắc (regex) thay vì phụ thuộc hoàn toàn vào một mô hình duy nhất.

---

## 7. AI log và điều đã học

### Bảng nhật ký dùng AI
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4: LLM API | "Cách cấu hình call_openai với GPT-6 model để không bị lỗi 400 và không bị tiêu tốn token cho reasoning?" | Hướng dẫn thiết lập `reasoning_effort="none"` và tham số `max_completion_tokens` thay vì `max_tokens` thông thường | Viết kiểm thử giả lập (mock unit test) để xác nhận chính xác các tham số được gửi vào client Chat Completions |
| Phần 4: Defended Prompt | "Thiết kế prompt phòng thủ chống Prompt Injection trong bài toán phân loại tin nhắn SMS" | Đề xuất phân tách rạch ròi chỉ dẫn hệ thống và dữ liệu người dùng thông qua thẻ XML `<sms>...</sms>`, hướng dẫn coi nội dung trong thẻ là dữ liệu không tin cậy | Thử nghiệm với các ca tấn công injection trong bộ 50 case khó, xác nhận mô hình phòng thủ thành công |

### Điều hiểu ra hôm nay mà trước đó chưa hiểu:
1. **Bản chất của các thế hệ AI:** Mỗi bước tiến từ Machine Learning, Deep Learning, Embedding cho đến LLM thực chất chỉ là sự dịch chuyển về việc *ai là người trích xuất đặc trưng* (người chọn $\rightarrow$ mạng tự học $\rightarrow$ mượn encoder pre-train $\rightarrow$ LLM đọc cả prompt).
2. **Đừng vội dùng mô hình phức tạp:** Deep Learning (MLP) không phải lúc nào cũng chiến thắng mô hình tuyến tính cổ điển nếu dữ liệu biểu diễn đầu vào chưa mang tính liên tục và ngữ cảnh sâu.
3. **Bài toán sản phẩm thật luôn là sự cân bằng đa chiều:** Điểm F1 cao chưa đủ để đưa vào sản phẩm; một kỹ sư AI thực thụ phải luôn tính toán đường Pareto giữa chất lượng, độ trễ p95, thông lượng và chi phí vận hành hàng tháng.
