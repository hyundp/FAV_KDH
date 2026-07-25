import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class DriftAgent(flax.struct.PyTreeNode):
    """DRIFT agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def get_drift(self, pos, gen):
        targets = jnp.concatenate([gen, pos], axis=0)
        G = gen.shape[0]

        dist = jnp.linalg.norm(gen[:, jnp.newaxis, :] - targets[jnp.newaxis, :, :], axis=-1)
        dist = dist.at[jnp.arange(G), jnp.arange(G)].set(1e6)
        kernel = jnp.exp(-dist / self.config['temp'])
        
        normalizer = jnp.sqrt(jnp.maximum(kernel.sum(axis=-1, keepdims=True) * kernel.sum(axis=0, keepdims=True), 1e-6))
        normalized_kernel = kernel / normalizer
        
        pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(axis=-1, keepdims=True)
        pos_V = pos_coeff @ targets[G:]
        neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(axis=-1, keepdims=True)
        neg_V = neg_coeff @ targets[:G]
        return pos_V - neg_V

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        gen_multiplier = self.config['gen_multiplier']
        rng, gen_rng = jax.random.split(rng, 2)

        obs_repeated = jnp.repeat(batch['observations'], gen_multiplier, axis=0)

        # 학습 시에는 grad_params를 사용해서 actor 파라미터로 gradient가 흐르도록 한다.
        noises = jax.random.normal(
            gen_rng,
            (
                *obs_repeated.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        gen = self.network.select('actor')(obs_repeated, noises, params=grad_params)
        gen = jnp.clip(gen, -1, 1)

        gen = gen.reshape(batch_size, gen_multiplier, action_dim)
        pos = batch['actions'][:, jnp.newaxis, :] 

        drift_fn = jax.vmap(self.get_drift) 
        drift = drift_fn(pos, gen)

        target_action = jax.lax.stop_gradient(gen + drift)
        drift_loss = jnp.mean(jnp.square(target_action - gen))
        
        return drift_loss, {
            'drift_loss': drift_loss,
            'drift_norm': jnp.linalg.norm(drift, axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = encoder_module()

        # Define networks.
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='drift',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            temp=1.0,  # Temperature for the drift kernel.
            gen_multiplier=8, 
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
