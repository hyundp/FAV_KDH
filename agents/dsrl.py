import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, LogParam, ActorVectorField, MLP, TanhNormal


class DSRLAgent(flax.struct.PyTreeNode):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        # Critic loss.
        rng, sample_rng, noise_rng = jax.random.split(rng, 3)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        # Latent critic distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch['actions'].shape[0], batch['actions'].shape[-1]))
        actions = self.sample_flow_actions(batch['observations'], noises=noises)
        actions = jnp.clip(actions, -1, 1)
        target_qs = self.network.select('critic')(batch['observations'], actions)
        qs = self.network.select('z_critic')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((qs - target_qs) ** 2)

        total_loss = critic_loss + distill_loss

        return total_loss, {
            'total_loss': total_loss,
            'critic_loss': critic_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape

        # BC flow loss.
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel))

        # Actor loss.
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.sample(seed=rng)
        log_probs = dist.log_prob(actions)
        actions = actions * self.config['noise_scale']

        qs = self.network.select('z_critic')(batch['observations'], actions)
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

        # Entropy loss.
        alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = flow_loss + actor_loss + alpha_loss

        action_std = dist._distribution.stddev()

        return total_loss, {
            'total_loss': total_loss,
            'flow_loss': flow_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': alpha,
            'entropy': -log_probs.mean(),
            'action_std': action_std.mean(),
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
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
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor_bc_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        action_dim = self.config['action_dim']
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)
        dist = self.network.select('actor')(observations)
        noises = dist.sample(seed=seed)
        noises = jnp.clip(noises, -1, 1)
        noises = noises * self.config['noise_scale']

        actions = self.sample_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        noises,
    ):
        actions = noises
        model = self.network.select('target_actor_bc_flow' if self.config['use_target_latent'] else 'actor_bc_flow')
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = model(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
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

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        critic_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config["value_layer_norm"], num_ensembles=config["num_qs"])
        actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            z_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_actions[..., :1])),
            target_actor_bc_flow=(copy.deepcopy(actor_bc_flow_def), (ex_observations, ex_actions, ex_actions[..., :1])),
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
        params['modules_target_actor_bc_flow'] = params['modules_actor_bc_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dsrl',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            num_qs=2,
            rho=0.0,
            discount=0.99,
            tau=0.005,
            flow_steps=10,
            best_of_n=1,
            noise_scale=1.0,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            use_target_latent=True,
        )
    )
    return config
