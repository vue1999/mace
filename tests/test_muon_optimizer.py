import torch

from mace.tools.optimizers import MuonWithAdam, build_muon_param_groups


def test_muon_optimizer_step_reduces_loss():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 3)

    param_groups = [
        {
            "params": list(model.parameters()),
            "lr": 0.02,
            "weight_decay": 0.0,
        }
    ]
    muon_groups = build_muon_param_groups(
        param_groups,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    optimizer = MuonWithAdam(muon_groups, ns_steps=2)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        baseline_loss = loss_fn(model(inputs), targets).item()

    optimizer.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        updated_loss = loss_fn(model(inputs), targets).item()

    assert updated_loss <= baseline_loss

