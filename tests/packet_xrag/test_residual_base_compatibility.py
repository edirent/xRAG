import pytest
import torch

from scripts.packet_xrag.composition_training_common import static2_k2_base


def test_static2_base_replicates_singleton_packet_deterministically():
    packet = torch.arange(2 * 4096, dtype=torch.float32).reshape(1, 2, 4096)
    base = static2_k2_base(packet)
    assert base.shape == (4, 4096)
    assert torch.equal(base[:2], base[2:])


def test_static2_base_preserves_first_two_packets():
    packets = torch.arange(3 * 2 * 4096, dtype=torch.float32).reshape(3, 2, 4096)
    assert torch.equal(static2_k2_base(packets), packets[:2].reshape(4, 4096))
    with pytest.raises(ValueError, match="at least one"):
        static2_k2_base(torch.empty(0, 2, 4096))
