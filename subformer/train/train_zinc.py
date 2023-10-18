import torch
from torch.optim import Adam, AdamW
from subformer.modules.model import SubFormer
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
from subformer.utils.transform import JunctionTree, get_transform_zinc
from subformer.utils.utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed(4321)

root = '../../dataset/zinc'

transform = get_transform_zinc(add_virtual_node=False)
train_dataset = ZINC(root, subset=True, split='train', pre_transform=transform)
val_dataset = ZINC(root, subset=True, split='val', pre_transform=transform)
test_dataset = ZINC(root, subset=True, split='test', pre_transform=transform)
train_loader = DataLoader(train_dataset, 64, shuffle=True)
val_loader = DataLoader(val_dataset, 1000, shuffle=False)
test_loader = DataLoader(test_dataset, 1000, shuffle=False)

epochs = 500
model = SubFormer(
    hidden_channels=64,
    out_channels=1,
    num_mp_layers=2,
    num_enc_layers=3,
    mp_dropout=0,
    mp_dropout_edge=0,
    enc_dropout=0.1,
    local_mp='gine',
    learn_gating=True,
    activation='relu',
    back_activation='leaky_relu',
    enc_activation='relu',
    aggregation='sum',
    pe_fea=False,
    pe_dim=10,
    n_head=8,
    padding_length=40,
    d_model=128,
    dim_feedforward=128,
    binary_readout=True,
    readout_channels=192,
    concat_pe=True,
    use_deg=True,
    use_lpe=False,
    use_spa=True,
    return_raw=False,
).to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of parameters: ', count_parameters(model))


def train(epoch):
    model.train()
    total_loss = 0

    for iter, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    total_error = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        total_error += (out.squeeze() - data.y).abs().sum().item()

    return total_error / len(loader.dataset)



test_maes = []
for run in range(1, 6):
    print()
    print(f'Run {run}:')
    print()

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.001, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=20, min_lr=0.00001)

    best_val_mae = test_mae = float('inf')
    for epoch in range(1, epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_mae = test(val_loader)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            test_mae = test(test_loader)

        print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
              f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')

    test_maes.append(test_mae)


test_mae = torch.tensor(test_maes)
print('===========================')
print(f'Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}')
