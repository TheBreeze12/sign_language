import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class Attention(nn.Cell):
    """注意力机制 — 对 BiLSTM 输出做加权聚合"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.dense1 = nn.Dense(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Dense(hidden_size, 1)

    def construct(self, lstm_output):
        # lstm_output: [B, Seq, Hidden*2]
        scores = self.dense2(self.tanh(self.dense1(lstm_output)))  # [B, Seq, 1]
        weights = ops.softmax(scores, axis=1)
        context = ops.reduce_sum(weights * lstm_output, axis=1)    # [B, Hidden*2]
        return context


class BiLSTMAttentionModel(nn.Cell):
    """BiLSTM + Attention 手语识别模型 (MindSpore 版)"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super(BiLSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm((hidden_size * 2,))
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Dense(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Dense(hidden_size, num_classes)

        # Get device target in __init__, not in construct
        device_target = ms.get_context("device_target")
        self.input_dtype = ms.float16 if device_target == "Ascend" else ms.float32

    def construct(self, x):
        # x: [B, Seq, InputSize]
        x = ops.cast(x, self.input_dtype)
        out, (_, _) = self.lstm(x)   # [B, Seq, Hidden*2]
        out = self.ln(out)
        context = self.attention(out)
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out