import jax
import jax.numpy as jnp
from typing import Any

PyTree = Any


def tree_stack(trees: list[PyTree]) -> PyTree:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree: PyTree) -> list[PyTree]:
    leaves, treedef = jax.tree.flatten(tree)
    n = leaves[0].shape[0] if leaves else 0
    return [treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(n)]


def normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def quat_to_rot_mat(quat: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    rot = jnp.stack([
        1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)
    ], axis=-1).reshape(*quat.shape[:-1], 3, 3)
    
    return rot


def quat_rotate_inverse(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    q_w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    
    a = vec * (2.0 * q_w ** 2 - 1.0)
    b = jnp.cross(q_vec, vec) * q_w * 2.0
    c = q_vec * jnp.sum(q_vec * vec, axis=-1, keepdims=True) * 2.0
    
    return a - b + c


def wrap_to_pi(angles: jnp.ndarray) -> jnp.ndarray:
    return (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi


def exp_kernel(x: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    return jnp.exp(-jnp.abs(x) / scale)


def exp_squared_kernel(x: jnp.ndarray, scale: float = 1.0) -> jnp.ndarray:
    return jnp.exp(-jnp.sum(x ** 2, axis=-1) / scale)

