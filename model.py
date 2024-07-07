import torch
import torch.nn as nn

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PricePredictor(nn.Module):
    def __init__(self) -> None:
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_0 = torch.zeros(2, x.size(0), 50).to(device)
        c_0 = torch.zeros(2, x.size(0), 50).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
