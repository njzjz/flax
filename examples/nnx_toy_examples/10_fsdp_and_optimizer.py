# %%
import dataclasses
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

from matplotlib import pyplot as plt
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import typing as tp

mesh = jax.sharding.Mesh(
  mesh_utils.create_device_mesh((2, 4)),
  ('data', 'model'),
)


def named_sharding(*names: str | None) -> NamedSharding:
  return NamedSharding(mesh, P(*names))


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  data: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(
  embed=None,
  mlp='model',
  data='data',
)


class MLP(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.w1 = nnx.Param(
      nnx.initializers.lecun_normal()(rngs.params(), (din, dmid)),
      sharding=mesh_rules('embed', 'mlp'),
    )
    self.b1 = nnx.Param(
      jnp.zeros((dmid,)),
      sharding=mesh_rules('mlp'),
    )
    self.w2 = nnx.Param(
      nnx.initializers.lecun_normal()(rngs.params(), (dmid, dout)),
      sharding=mesh_rules('embed', 'mlp'),
    )

  def __call__(self, x: jax.Array):
    return nnx.relu(x @ self.w1 + self.b1) @ self.w2

# TODO(cgarciae): upstream as nnx.variables
def nnx_variables(node):
  flat_variables = {}
  for path, value in nnx.iter_graph(node):
    if isinstance(value, nnx.Variable):
      flat_variables[path] = value
  return nnx.State.from_flat_path(flat_variables)

# TODO(cgarciae): upstream as Variable.get_metadata
def variable_get_metadata(variable: nnx.Variable):
  metadata = vars(variable).copy()
  del metadata['raw_value']
  del metadata['_trace_state']
  return metadata


class SGDState(nnx.Variable):
  pass


class SGD(nnx.Object):
  def __init__(self, model, lr, momentum=0.9):
    def init_optimizer_state(variable: nnx.Variable):
      return SGDState(
        jnp.zeros_like(variable.value), **variable_get_metadata(variable)
      )

    self.lr = lr
    self.params = nnx_variables(model)
    self.momentum = jax.tree.map(init_optimizer_state, self.params)
    self.decay = momentum

  def update(self, grads: nnx.State):
    def update_fn(
      params: nnx.Variable, momentum: SGDState, grad: nnx.VariableState
    ):
      # v_t = β * v_{t-1} + (1 - β) * ∇J(θ_t)
      momentum.value = self.decay * momentum + (1 - self.decay) * grad.value
      # θ_{t+1} = θ_t - α * v_t
      params.value -= self.lr * momentum

    jax.tree.map(update_fn, self.params, self.momentum, grads)

# TODO(cgarciae): upstream as State.map
def state_map(
  state: nnx.State, f: tp.Callable[[tuple, nnx.VariableState], nnx.VariableState]
):
  flat_state = state.flat_state()
  for path, variable_state in flat_state.items():
    flat_state[path] = f(path, variable_state)
  return nnx.State.from_flat_path(flat_state)


@nnx.jit
def create_model():
  model = MLP(1, 32, 1, rngs=nnx.Rngs(0))
  optimizer = SGD(model, 0.01, momentum=0.9)
  state = nnx.state(optimizer)
  sharded_state = jax.lax.with_sharding_constraint(
    state, nnx.get_named_sharding(state, mesh)
  )

  def get_named_shardings(path: tuple, value: nnx.VariableState):
    if path[0] == 'params':
      return value.replace(NamedSharding(mesh, P(*value.sharding)))
    elif path[0] == 'momentum':
      # currently the same as above but in general it could be different
      return value.replace(NamedSharding(mesh, P(*value.sharding)))
    else:
      raise ValueError(f'Unknown path: {path}')

  named_shardings = state_map(state, get_named_shardings)
  sharded_state = jax.lax.with_sharding_constraint(state, named_shardings)
  nnx.update(optimizer, sharded_state)
  return model, optimizer


model, optimizer = create_model()

jax.debug.visualize_array_sharding(model.w1.value)
jax.debug.visualize_array_sharding(optimizer.momentum.w1.value)


@nnx.jit
def train_step(model: MLP, optimizer: SGD, x, y):
  def loss_fn(model):
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    return loss

  loss, grad = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grad)
  return loss


X = np.linspace(-2, 2, 100)[:, None]
Y = 0.8 * X**2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size, num_steps):
  for _ in range(num_steps):
    idx = np.random.choice(len(X), size=batch_size)
    yield X[idx], Y[idx]


losses = []
for step, (x_batch, y_batch) in enumerate(
  dataset(batch_size=32, num_steps=10_000)
):
  x_batch, y_batch = jax.device_put((x_batch, y_batch), named_sharding('data'))
  loss = train_step(model, optimizer, x_batch, y_batch)
  losses.append(float(loss))
  if step % 1000 == 0:
    print(f'Step {step}: Loss = {loss}')

plt.figure()
plt.plot(losses[20:])

y_pred = model(X)
plt.figure()
plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()
