import torch
import numpy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(mask, label, threshold=0.5, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == 'binary':
            score = label
        else:
            score = (torch.mean(mask, axis=(1, 2, 3)) + label) / 2

        preds = (score > threshold).to(device)

        return preds, score


def test_accuracy(model, test_dl):
    acc = 0
    total = len(test_dl.dataset)
    for img, mask, label in test_dl:
        img = img.to(device)
        label = label
        net_mask, net_label = model(img)
        preds, _ = predict(net_mask, net_label)
        ac = (preds.cpu() == label.cpu())
        acc += torch.sum(ac).item()
    return (acc / total) * 100


def test_loss(model, test_dl, loss_fn):
    loss = 0
    total = len(test_dl)
    for img, mask, label in test_dl:
        img = img.to(device)
        mask = mask.to(device)
        label = label.to(device)
        net_mask, net_label = model(img)
        losses = loss_fn(net_mask, net_label, mask, label)
        loss += torch.mean(losses).item()
    return loss / total
    