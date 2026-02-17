import torch
from snapuq.models.vision.resnet_cifar import resnet20_cifar
from snapuq.snapuq import SnapUQ

def test_snapuq_forward_shapes():
    backbone = resnet20_cifar(num_classes=10)
    name_to_module = dict(backbone.named_modules())
    taps = [name_to_module["conv1"], name_to_module["layer1.2"], name_to_module["layer2.2"], name_to_module["layer3.2"]]
    suq = SnapUQ(backbone, taps, rank=8)
    suq.infer_dims(torch.zeros(1,3,32,32))
    out = suq(torch.zeros(4,3,32,32), return_uq=True)
    assert out.logits.shape == (4,10)
    assert out.S.shape == (4,)
    assert out.m.shape == (4,)
    assert len(out.tap_surprisals) == 3
