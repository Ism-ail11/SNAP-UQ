import torch
from snapuq.models.vision.lenet import LeNetMNIST
from snapuq.snapuq import SnapUQ

def test_nll_decreases_basic_step():
    backbone = LeNetMNIST(num_classes=10)
    name_to_module = dict(backbone.named_modules())
    taps = [name_to_module["features.0"], name_to_module["features.3"], name_to_module["features.6"], name_to_module["classifier.1"]]
    suq = SnapUQ(backbone, taps, rank=4)
    suq.infer_dims(torch.zeros(1,1,28,28))
    # freeze backbone
    for p in suq.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(suq.predictors.parameters(), lr=1e-2)
    x = torch.randn(16,1,28,28)
    loss0 = float(suq.gaussian_nll_loss(x).item())
    for _ in range(5):
        opt.zero_grad()
        loss = suq.gaussian_nll_loss(x)
        loss.backward()
        opt.step()
    loss1 = float(suq.gaussian_nll_loss(x).item())
    assert loss1 <= loss0 + 1e-4  # should not blow up
