from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _device_put_replicated(x, devices):
    """Drop-in replacement from the JAX pmap migration guide: stack one copy per
    device along a new leading axis and shard that axis across the devices."""
    sharding = NamedSharding(Mesh(np.array(devices), ("replica",)), PartitionSpec("replica"))
    return jax.tree.map(
        lambda leaf: jax.device_put(jnp.stack([jnp.asarray(leaf)] * len(devices)), sharding), x
    )


def install_brax_compat() -> None:
    """Brax 0.14.2 PPO still calls jax.device_put_replicated, removed in JAX 0.10."""
    if not hasattr(jax, "device_put_replicated"):
        jax.device_put_replicated = _device_put_replicated
