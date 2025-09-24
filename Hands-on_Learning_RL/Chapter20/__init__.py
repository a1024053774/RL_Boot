import logging

from gym import envs
from gym.envs.registration import register

logger = logging.getLogger(__name__)


'''
对ma-gym的github代码的修复版本
该文件目录是ma-gym/ma_gym/__init__.py
'''

# Register openai's environments as multi agent (best-effort, version compatible)
# Some newer gym/gymnasium versions changed internal EnvSpec fields:
# - Older: spec._kwargs
# - Newer: spec.kwargs
# Also registry structure differs (may have .all(), .values(), or be a dict)
try:
    env_specs = []
    registry = getattr(envs, 'registry', None)
    if registry is not None:
        if hasattr(registry, 'all'):
            iterable = registry.all()
        elif hasattr(registry, 'values'):
            iterable = registry.values()
        elif isinstance(registry, dict):  # gymnasium>=0.29 style
            iterable = registry.values()
        else:
            iterable = []
        for spec in iterable:
            entry_point = getattr(spec, 'entry_point', '')
            if isinstance(entry_point, str) and 'gym.envs' in entry_point:
                env_specs.append(spec)
    for spec in env_specs:
        base_kwargs = {}
        if hasattr(spec, 'kwargs') and isinstance(getattr(spec, 'kwargs'), dict):
            base_kwargs.update(spec.kwargs)  # public attr (newer gym)
        elif hasattr(spec, '_kwargs') and isinstance(getattr(spec, '_kwargs'), dict):
            base_kwargs.update(spec._kwargs)  # legacy private attr
        # Ensure name passed for wrapper
        register(
            id='ma_' + spec.id,
            entry_point='ma_gym.envs.openai:MultiAgentWrapper',
            kwargs={'name': spec.id, **base_kwargs}
        )
except Exception as e:
    logger.warning(f"Skipping automatic OpenAI Gym env registration due to: {e}")

# add new environments : iterate over full observability
for i, observability in enumerate([False, True]):
    register(
        id='CrossOver-v' + str(i),
        entry_point='ma_gym.envs.crossover:CrossOver',
        kwargs={'full_observable': observability, 'step_cost': -0.5}
    )

    register(
        id='Checkers-v' + str(i),
        entry_point='ma_gym.envs.checkers:Checkers',
        kwargs={'full_observable': observability}
    )

    register(
        id='Switch2-v' + str(i),
        entry_point='ma_gym.envs.switch:Switch',
        kwargs={'n_agents': 2, 'full_observable': observability, 'step_cost': -0.1}
    )
    register(
        id='Switch4-v' + str(i),
        entry_point='ma_gym.envs.switch:Switch',
        kwargs={'n_agents': 4, 'full_observable': observability, 'step_cost': -0.1}
    )

    register(
        id='TrafficJunction-v' + str(i),
        entry_point='ma_gym.envs.traffic_junction:TrafficJunction',
        kwargs={'full_observable': observability}
    )

    register(
        id='Lumberjacks-v' + str(i),
        entry_point='ma_gym.envs.lumberjacks:Lumberjacks',
        kwargs={'full_observable': observability}
    )


register(
    id='Combat-v0',
    entry_point='ma_gym.envs.combat:Combat',
)
register(
    id='PongDuel-v0',
    entry_point='ma_gym.envs.pong_duel:PongDuel',
)

for game_info in [[(5, 5), 2, 1], [(7, 7), 4, 2]]:  # [(grid_shape, predator_n, prey_n),..]
    grid_shape, n_agents, n_preys = game_info
    _game_name = 'PredatorPrey{}x{}'.format(grid_shape[0], grid_shape[1])
    register(
        id='{}-v0'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys
        }
    )
    # fully -observable ( each agent sees observation of other agents)
    register(
        id='{}-v1'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True
        }
    )

    # prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v2'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )

    # full observability + prey is initialized at random location and thereafter doesn't move
    register(
        id='{}-v3'.format(_game_name),
        entry_point='ma_gym.envs.predator_prey:PredatorPrey',
        kwargs={
            'grid_shape': grid_shape, 'n_agents': n_agents, 'n_preys': n_preys, 'full_observable': True,
            'prey_move_probs': [0, 0, 0, 0, 1]
        }
    )
