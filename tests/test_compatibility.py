"""Permanent regression checks for Python 3.11 / DGL 1.1 compatibility."""
import torch
import dgl
import json

from models.training.utils import flatten_dict
from models.zero_shot_models.message_aggregators.mscn import MscnConv


def test_nested_training_configuration():
    assert flatten_dict({"optimizer": {"lr": 0.001}, "seed": 0}) == {
        "optimizer_lr": 0.001, "seed": 0
    }


def test_messages_sum_children_and_support_backpropagation():
    graph = dgl.heterograph({("child", "to", "parent"): ([0, 1], [0, 0])})
    child = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    parent = torch.tensor([[10., 20.]], requires_grad=True)
    conv = MscnConv(hidden_dim=2, test=True, n_layers=0, width_factor=1, residual=False,
                    activation_class_name="LeakyReLU", activation_class_kwargs={},
                    norm_class_name="Identity", norm_class_kwargs={},
                    activation=False, dropout=False, p_dropout=0)
    result = conv(graph=graph, etypes=graph.canonical_etypes,
                  out_node_types=["parent"], feat_dict={"child": child, "parent": parent})
    torch.testing.assert_close(result["parent"], torch.tensor([[14., 26.]]))
    result["parent"].sum().backward()
    torch.testing.assert_close(child.grad, torch.ones_like(child))


def test_test_workload_order_does_not_depend_on_training_rng(tmp_path, monkeypatch):
    from models.dataset.dataset_creation import create_dataloader, plan_collator_dict
    source = tmp_path / "workload.json"
    source.write_text(json.dumps({"database_stats": {}, "run_kwargs": {},
                                 "parsed_plans": [{"plan_runtime": 1000 + i} for i in range(20)]}))
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    monkeypatch.setitem(plan_collator_dict, "test", lambda batch, **kwargs: [i for i, plan in batch])
    _, _, _, _, loaders = create_dataloader([str(source)], [str(source)], str(stats), None, "test",
                                             batch_size=4, num_workers=0, loss_class_name="QLoss")
    first = [i for batch in loaders[0] for i in batch]
    torch.rand(100)
    second = [i for batch in loaders[0] for i in batch]
    assert first == second == list(range(20))
