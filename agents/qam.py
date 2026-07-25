import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class QAMAgent(flax.struct.PyTreeNode):
    """Q-learning with adjoint matching (QAM), adapted for LiFA.

    Modified from https://github.com/ColinQiyangLi/qam/blob/main/agents/qam.py  
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        next_actions = self.sample_actions(batch['next_observations'], seed=rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    @partial(jax.jit, static_argnames=('flow_steps',))
    def adj_matching(self, obs, rng, flow_steps=None):
        flow_steps = self.config['flow_steps'] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim']
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select('target_actor_slow' if self.config['target_actor'] else 'actor_slow')

        h = 1 / flow_steps
        xs = [x]
        ts = []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config['residual']:
                    v = self.network.select('actor_fast')(obs, x, t) + actor_slow(obs, x, t)
                else:
                    v = self.network.select('actor_fast')(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        critic_network = 'target_critic' if self.config['use_target_grad'] else 'critic'
        if self.config['clip_adj']:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
        else:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, y).mean(axis=0).sum(), 1)

        adj = -grad_fn(obs, xs[-1]) * self.config['inv_temp']
        pre_adj_info = {
            'adj_max': jnp.abs(adj).max(),
            'adj_std': jnp.abs(adj).std(),
            'adj_mean': jnp.abs(adj).mean(),
        }
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            adjs.append(adj)

        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, adj_rng, edit_rng = jax.random.split(rng, 5)

        # BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean((pred - vel) ** 2)
        actor_loss = flow_loss

        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select('target_actor_slow' if self.config['target_actor'] else 'actor_slow')

        # Adjoint-matching.
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch['observations'], adj_rng)
        h = 1 / self.config['flow_steps']
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch['observations'][None], self.config['flow_steps'], axis=0)
        vf_fine = self.network.select('actor_fast')(observations, xs, ts, params=grad_params)
        vf_base = actor_slow(observations, xs, ts)

        if self.config['residual']:
            adj_loss = jnp.sum(jnp.square(vf_fine * 2 / sigmas + sigmas * adjs), axis=-1)
        else:
            adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)

        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))
        info['adj_loss'] = adj_loss
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        if self.config['fql_alpha'] > 0.:
            fql_noises = jax.random.normal(edit_rng, (batch_size, action_dim))
            flow_actions = self.compute_flow_actions(
                batch['observations'], fql_noises,
                model='slow,fast' if self.config['residual'] else 'fast')

            os_actions = self.network.select('one_step_actor')(
                batch['observations'], fql_noises, params=grad_params)
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)

            os_actions = jnp.clip(os_actions, -1, 1)
            fql_qs = self.network.select('critic')(batch['observations'], actions=os_actions)
            fql_q = jnp.mean(fql_qs, axis=0)
            fql_q_loss = -fql_q.mean()

            info['fql_distill_loss'] = fql_distill_loss
            info['fql_q_loss'] = fql_q_loss
            actor_loss += fql_q_loss + fql_distill_loss * self.config['fql_alpha']

        return actor_loss + total_fast_loss, {'flow_loss': flow_loss, 'fast_loss': total_fast_loss, **info}

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
        self.target_update(new_network, 'actor_slow')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        rng, edit_rng = jax.random.split(seed)

        action_dim = self.config['action_dim']
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[:-len(self.config['ob_dims'])],
                self.config['best_of_n'], action_dim,
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config['best_of_n'], axis=-2)

        if self.config['fql_alpha'] > 0.:
            actions = self.network.select('one_step_actor')(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:
            if self.config['inv_temp'] == 0.:
                actions = self.compute_flow_actions(observations, noises, model='slow')
            else:
                actions = self.compute_flow_actions(observations, noises, model='slow,fast' if self.config['residual'] else 'fast')
            actions = jnp.clip(actions, -1, 1)

        # best-of-n sampling
        q = self.network.select('critic')(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config['best_of_n'], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @partial(jax.jit, static_argnames='model')
    def compute_flow_actions(self, observations, noises, model='slow'):
        actions = noises
        networks = [self.network.select(f'actor_{m}') for m in model.split(',')]

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])
            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, ex_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, ex_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, ex_actions, ex_times)),
        )

        if config['fql_alpha'] > 0.:
            network_info['one_step_actor'] = (copy.deepcopy(actor_def), (ex_observations, ex_actions, None))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['clip_grad']:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),
                                     optax.adam(learning_rate=config['lr']))
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qam',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            q_agg='mean',
            flow_steps=10,
            best_of_n=1,
            inv_temp=0.3,
            fql_alpha=0.,
            target_actor=True,
            residual=False,
            clip_adj=True,
            use_target_grad=True,
            clip_grad=True,
        )
    )
    return config
