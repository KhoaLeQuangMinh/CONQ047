# Stock Price Forecasting using NLinear on Log-Return

## Tổng Quan

Mô hình này dự đoán giá cổ phiếu trong tương lai dựa trên **log-return** thay vì giá đóng cửa trực tiếp. Sử dụng **2 mô hình NLinear** để kết hợp thông tin dài hạn (global trend) và ngắn hạn (local trend), áp dụng **sliding window ensemble** và **post-processing** để cải thiện độ mượt và độ chính xác dự đoán.

---

## 1. Thư viện & Cấu Hình Môi Trường

* **Python 3.10+**
* **Pytorch**
* **NumPy, Pandas, Matplotlib**
* GPU được ưu tiên nếu có (CUDA).

Cấu hình **seed** để đảm bảo kết quả có thể tái lập (`seed_everything(42)`).

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 2. Tiền Xử Lý Dữ Liệu

1. Load dữ liệu từ CSV, sắp xếp theo thời gian.
2. Tính **log-price**:

```python
df['close_log'] = np.log(df['close'] + 1e-8)
```

3. Tính **log-return** làm target:

```python
df['daily_return_log'] = df['close_log'].diff().fillna(0)
```

4. Chỉ giữ cột target để training:

```python
df_processed = df[['daily_return_log']].copy()
```

---

## 3. Mô Hình NLinear

* Mỗi model là **Linear Layer** đơn giản nhưng hoạt động trên toàn bộ sequence (seq → pred).
* Input là `seq_len` log-return liên tiếp, output là `pred_len` log-return tiếp theo.
* Chuẩn hóa sequence bằng cách trừ đi giá trị cuối cùng (`last_value`) trước khi đưa vào linear layer để model học **delta log-return**.

```python
pred = self.linear(x - last_value) + last_value
```

---

## 4. Dataset & Dataloader

* Sequence dataset: `seq_len` log-return → predict `pred_len` log-return tiếp theo.
* Dataloader shuffle dữ liệu trong training để tăng khả năng generalize, nhưng từng sequence vẫn giữ trật tự thời gian.

---

## 5. Training

* Loss: **MSELoss** trên log-return.
* Optimizer: **Adam**.
* Scheduler: **StepLR** giảm lr 50% mỗi 20 epoch.
* Train từng model riêng biệt: **Long-Term** và **Short-Term**.

```python
model_long = train_model(NLinear(L1, T), L1, T, NUM_EPOCHS, BATCH_SIZE)
model_short = train_model(NLinear(L2, T), L2, T, NUM_EPOCHS, BATCH_SIZE)
```

---

## 6. Sliding Window Ensemble (Trong Dự Đoán)

* Lấy nhiều **window trượt từ lịch sử** làm input → tạo nhiều dự đoán cho cùng `pred_len`.
* Kết hợp các dự đoán bằng **weighted average**, trọng số cao hơn cho window gần hơn thời điểm dự đoán.

```python
final_predictions_return = np.sum(predictions_matrix * weights[:, np.newaxis], axis=0)
```

* Thao tác này **ensemble dự đoán gần nhau** trong cùng 1 model.

---

## 7. Meta-Ensemble Giữa 2 Model

* Sau khi có dự đoán từ **Long-Term** và **Short-Term**, kết hợp theo trọng số:

```python
final_predictions_return_raw = WEIGHT_LONG * pred_return_long + WEIGHT_SHORT * pred_return_short
```

* Trọng số ưu tiên **Short-Term** (local trend) hơn Long-Term (global trend).

---

## 8. Post-Processing

1. **EMA smoothing** để làm mượt các biến động quá nhỏ hoặc nhiễu:

```python
forecast_return_smooth = apply_ema(final_predictions_return_raw, ALPHA_EMA)
```

2. **Thêm nhiễu Gaussian nhỏ** để mô phỏng biến động thị trường:

```python
noise_base = np.random.normal(0, STOCHASTIC_FACTOR, size=T)
forecast_return_final = forecast_return_smooth + noise_base
```

---

## 9. Chuyển Log-Return → Giá Close

* Từ log-return dự đoán → log-price dự đoán bằng cách cộng dồn từ giá cuối cùng trong lịch sử:

```python
forecast_log_price = last_close_log + np.cumsum(forecast_return_final)
forecast_price_final = np.exp(forecast_log_price)
```

---

## 10. Lưu & Visualization

* Xuất kết quả ra CSV:

```python
submission = pd.DataFrame({'id': range(1, TOTAL_PREDICT_DAYS + 1), 'close': forecast_price_final})
submission.to_csv('submission_final_log_return.csv', index=False)
```

* Vẽ đồ thị kết hợp **lịch sử + dự đoán**, đánh dấu ranh giới giữa quá khứ và tương lai.

---

## 11. Tổng Kết Pipeline

1. Load & tiền xử lý dữ liệu → log-price, log-return
2. Khởi tạo 2 model NLinear → Long-Term & Short-Term
3. Chuẩn bị dataset & dataloader
4. Training từng model với MSE trên log-return
5. Dự đoán → Sliding Window Ensemble trong từng model
6. Meta-Ensemble giữa 2 model (trọng số Long/Short)
7. Post-Processing → EMA + Stochastic Noise
8. Chuyển log-return → log-price → price
9. Lưu kết quả & visualization

---

## 12. Thông Số Chính

| Tham số                    | Ý nghĩa                                |
| -------------------------- | -------------------------------------- |
| L1 / L2                    | Sequence length Long-Term / Short-Term |
| K1 / K2                    | Số window trượt cho ensemble           |
| S1 / S2                    | Slide step cho ensemble                |
| T                          | Số ngày dự đoán (output length)        |
| ALPHA_EMA                  | Hệ số làm mượt EMA                     |
| STOCHASTIC_FACTOR          | Biên độ thêm nhiễu Gaussian            |
| WEIGHT_LONG / WEIGHT_SHORT | Trọng s                                |
