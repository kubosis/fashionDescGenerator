import copy

import torch
import torch.nn as nn
import torchmetrics

CLIP_NORM = 1.0

def test(model: torch.nn.Module,
         data_loader: torch.utils.data.DataLoader,
         device: str,
         epoch: int | None =None):
    loss_fn = nn.CrossEntropyLoss()
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    model.eval()
    acc.reset()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            acc.update(y_hat, y)
            total_loss += loss_fn(y_hat, y)
    total_loss /= len(data_loader)
    return acc.compute(), total_loss

def train(model: nn.Module,
          optim: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          device: str='cpu',
          epochs: int=50,
          clip_norm: float=1.0):
    loss_fn = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
    best_model = None
    best_test_acc = 0.
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        current_lr = optim.param_groups[0]['lr']
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optim.step()
            scheduler.step()
            total_loss += loss.item()
            accuracy.update(y_hat, y)
        test_acc, test_loss = test(model, test_loader, device, epoch=epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())
        total_loss = total_loss / len(train_loader)
        out_epoch = epoch + 1
        out_epoch = " " + str(out_epoch) if out_epoch < 10 else str(out_epoch)
        print(f"[EPOCH {out_epoch}/{epochs}] || lr={current_lr:.2e}, Train loss: {total_loss:.3f}; Train acc: {accuracy.compute():.3f}; Test loss: {test_loss:.3f}; Test acc: {test_acc:.3f}")
        accuracy.reset()
    model.load_state_dict(best_model)
    return model