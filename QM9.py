import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet, global_mean_pool
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'data/QM9'
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 200
PATIENCE = 20  # for early stopping
TARGET_INDEX = 10  # QM9 targets: G (free energy) at 298.15 K

# Ensure data directory exists
os.makedirs(DATA_ROOT, exist_ok=True)

# 1. Load QM9 dataset
print('Loading QM9 dataset...')
dataset = QM9(DATA_ROOT)
# Filter out incomplete entries
dataset = dataset.shuffle()
dataset.data.y = dataset.data.y[:, TARGET_INDEX]

# 2. Split: 80% train, 10% val, 10% test
total = len(dataset)
train_len = int(0.8 * total)
val_len = int(0.1 * total)
test_len = total - train_len - val_len
train_set = dataset[:train_len]
val_set = dataset[train_len:train_len + val_len]
test_set = dataset[train_len + val_len:]

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# 3. Model Definition using SchNet
class FreeEnergyPredictor(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0):
        super().__init__()
        self.model = SchNet(hidden_channels=hidden_channels,
                            num_filters=num_filters,
                            num_interactions=num_interactions,
                            num_gaussians=num_gaussians,
                            cutoff=cutoff,
                            readout="add")
        # Final linear layer to predict a single scalar
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = self.model(data.z, data.pos, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.lin(x).view(-1)

# 4. Training and Evaluation

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data)
        loss = criterion(pred, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# 5. Initialize
model = FreeEnergyPredictor().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 6. Training Loop with Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print('Early stopping triggered!')
            break

# 7. Test Evaluation
model.load_state_dict(torch.load('best_model.pth'))
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.6f}')

# 8. Save final model and configs
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': {
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'hidden_channels': 128,
        'num_interactions': 6,
        'num_gaussians': 50
    }
}, 'final_checkpoint.pth')
