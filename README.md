# causality-benefit-in-control

Core code for a reinforcement-learning cooling-setpoint controller for a
residential unit simulated in EnergyPlus. The controller is trained against an
occupant who overrides the setpoint when they become uncomfortable, so the
learned policy trades electricity cost against the overrides it provokes. The
occupant driving the simulation is either the rule-based model in
`01_ground_truth_occupant.py` or, in the two arms compared in the paper, a
data-driven model learned from its behaviour.

Files are numbered in reading order. `rl_env.py` and `occ_model_collections.py`
keep plain names because they are imported as modules rather than run.

## Paper

This repository holds the code behind the following paper, and is the version
the reported results were produced with:

> *Applied Energy* (2026).
> <https://www.sciencedirect.com/science/article/pii/S0306261926014571>

## Files

- **`00_building_model.idf`** — the EnergyPlus building model: one residential
  unit with ideal-load cooling, whose setpoint schedule is what the agent
  actuates.
- **`01_ground_truth_occupant.py`** — the rule-based occupant, standalone.
  Fanger's heat balance gives a thermal load; a logistic model turns accumulated
  discomfort into setpoint overrides. `rl_env.py` carries its own copy of these
  two classes, so this file is a readable reference rather than an import.
- **`02_controller_training.py`** — PPO training entry point
  (Stable-Baselines3), vectorised over parallel EnergyPlus instances.
  `--ob_model_causality` picks which occupant drives the simulation: `0` the
  non-causal data-driven model, `4` the causal one, `-1` the rule-based occupant
  with no checkpoint.
- **`rl_env.py`** — the Gymnasium environment wrapping EnergyPlus: runs the
  simulation on its own thread, exchanges observations and actions once per zone
  timestep, and computes the reward from electricity cost and override count.
  Derived in part from `airboxlab/rllib-energyplus` — see the licence below.
- **`occ_model_collections.py`** — the data-driven occupant model:
  `LSTMOrdinalLogisticRegression` plus the sequence preparation used to train
  it. When one of its checkpoints is loaded it replaces the rule-based occupant
  and decides each override itself.
- **`toronto.epw`** — TMYx weather file for Toronto, read by EnergyPlus.
- **`weather_toronto.csv`** — the same year at 5-minute resolution, used to give
  the agent one- and two-hour-ahead outdoor conditions.

## Third-party licenses

### airboxlab/rllib-energyplus

The EnergyPlus ↔ RL interface in this repository is derived from
[airboxlab/rllib-energyplus](https://github.com/airboxlab/rllib-energyplus),
file `rleplus/env/energyplus.py`, which is distributed under the MIT License.

The derived file is `rl_env.py`, which carries a pointer to this notice in its
header. What it takes from upstream is the threading/queue harness that drives
EnergyPlus from a Gymnasium environment:

- `RunnerConfig` — the variables/meters/actuators configuration dataclass
- `EnergyPlusRunner` — `start`, `stop`, `failed`, `make_eplus_args`,
  `init_exchange`, `_init_callback`, `_init_handles`, `_flush_queues`, and the
  `obs_queue`/`act_queue` producer-consumer rendezvous registered on
  `callback_begin_zone_timestep_after_init_heat_balance`
- `EnergyPlusEnv` — the `gym.Env` skeleton and its abstract
  `get_variables` / `get_meters` / `get_actuators` / `get_observation_space` /
  `get_action_space` / `compute_reward` interface

### License text

```
MIT License

Copyright (c) 2022 Antoine Galataud

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
