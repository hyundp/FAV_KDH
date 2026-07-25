import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, LogParam, MLP
from utils.networks import TanhNormal


class RLPDAgent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent for offline RL.

    This agent can also be used for reinforcement learning with prior data (RLPD).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)

        next_dist = self.network.select('actor')(batch['next_observations'])
        next_actions = next_dist.sample(seed=sample_rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        next_q = next_qs.mean(axis=0) - next_qs.std(axis=0) * self.config["rho"]

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)

        # Actor loss.
        qs = self.network.select('critic')(batch['observations'], actions)
        q = jnp.mean(qs, axis=0) - jnp.std(qs, axis=0) * self.config["rho"]

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        # BC loss.
        bc_loss = -dist.log_prob(jnp.clip(batch['actions'], -1 + 1e-5, 1 - 1e-5)).mean() * self.config["bc_alpha"]
        total_loss = actor_loss + alpha_loss + bc_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'bc_loss': bc_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=seed)
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )

        actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)

        alpha_def = LogParam(init_value=config["init_temp"])

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rlpd',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            actor_layer_norm=False,
            value_hidden_dims=(256, 256),
            value_layer_norm=True,
            num_qs=2,
            rho=0.5,
            discount=0.99,
            tau=0.005,
            bc_alpha=0.0,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            init_temp=1.0,
        )
    )
    return config
