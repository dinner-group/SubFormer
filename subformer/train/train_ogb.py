import torch
from torch.optim import Adam, AdamW
from subformer.modules.model import SubFormer
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from subformer.utils.transform import get_transform
from subformer.utils.utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed(4321)

transform = get_transform(add_virtual_node=False)
ds = PygGraphPropPredDataset('ogbg-moltox21',
                             '../../dataset',
                             pre_transform=transform).shuffle()
evaluator = Evaluator(name='ogbg-moltox21')

idxs = []
data_list = []
for i, data in enumerate(ds):
    if data.x_clique.shape[0] < 1:
        idxs.append(i)
    else:
        data_list.append(data)
print('num of exclude: ', len(idxs))
dataset = data_list
num_train, num_trainval = round(0.8 * len(dataset)), round(0.9 * len(dataset))

train_dataset = dataset[:num_train]
val_dataset = dataset[num_train:num_trainval]
test_dataset = dataset[num_trainval:]
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, 512, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, 512, shuffle=False, num_workers=0)

epochs = 100
model = SubFormer(
    hidden_channels=256,
    out_channels=ds.num_tasks,
    num_mp_layers=3,
    num_enc_layers=4,
    mp_dropout=0.2,
    mp_dropout_edge=0.2,
    enc_dropout=0.5,
    local_mp='gine',
    learn_gating=True,
    activation='relu',
    back_activation='leaky_relu',
    enc_activation='relu',
    aggregation='sum',
    pe_fea=True,
    pe_dim=10,
    n_head=8,
    padding_length=120,
    d_model=512,
    dim_feedforward=1024,
    binary_readout=True,
    readout_channels=768,
    concat_pe=True,
    use_deg=True,
    use_lpe=False,
    use_spa=True,
).to(device)

print(model)
model.reset_parameters()
optimizer = AdamW(model.parameters(), lr=0.0005, amsgrad=True)


def train(epoch):
    model.train()

    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        out = model(data)[mask]
        y = data.y.to(torch.float)[mask]

        loss = torch.nn.BCEWithLogitsLoss()(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data))
        print(model(data))
        y_trues.append(data.y)
        print(data.y)
    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]


for run in range(1, 6):
    print()
    print(f'Run {run}:')
    print()

    model.reset_parameters()
    optimizer = AdamW(model.parameters(), lr=0.0005, amsgrad=True, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_perf = test_perf = 0
    for epoch in range(1, epochs + 1):
        loss = train(epoch)
        train_perf = test(train_loader)
        val_perf = test(val_loader)
        scheduler.step(val_perf)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = test(test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_perf:.4f}, Val: {val_perf:.4f}, '
              f'Test: {test_perf:.4f}')
