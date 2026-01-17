import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=128, output_dim=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        output = self.fc(lstm_out)
        return self.sigmoid(output)

def get_lstm_model():
    return LSTMModel()