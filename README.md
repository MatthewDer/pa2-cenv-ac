# 6x6 Checkers

A two-player 6x6 checkers game following the PettingZoo AEC API.

| Import | `from mycheckersenv import env` |
| --- | --- |
| Actions | Discrete |
| Agents | `agents = ['player_0', 'player_1']` |
| Agents | 2 |
| Action Shape | (1,) |
| Action Values | Discrete(144) |
| Observation Shape | (36,) |
| Observation Values | [0, 4] |

player_0 starts at the bottom of the board (rows 4–5) and moves upward. player_1 starts at the top (rows 0–1) and moves downward. Agents take turns alternately. The game ends when one player captures all opponent pieces, blocks them from moving, or the no-progress limit is reached.

## Observation Space

Each agent receives a flat array of shape `(36,)` representing the full 6x6 board in a player-relative orientation. The board is always presented so that the observing agent's pieces appear at the bottom (rows 4–5) regardless of which color they are playing. This is achieved by flipping the board vertically for player_1 and remapping piece values accordingly.

Cell values:

| Value | Meaning |
| --- | --- |
| 0 | Empty |
| 1 | My man |
| 2 | My king |
| 3 | Opponent's man |
| 4 | Opponent's king |

### Action Mask

The legal moves available to the current agent are provided in `infos[agent]["action_mask"]`, a binary array of shape `(144,)` where 1 indicates a legal action and 0 indicates an illegal one. If any jump (capture) is available, all simple move slots are masked out — captures are mandatory. During a multi-jump chain, only continuation jumps from the current piece's square are legal.

## Action Space

The action space is `Discrete(144)`. Actions are encoded as `(from_square * 4 + direction)` where direction is one of: 0=up-left, 1=up-right, 2=down-left, 3=down-right.

- Actions 0–71 are simple (non-capturing) moves
- Actions 72–143 are jump (capturing) moves

The 18 playable squares are the dark squares of the 6x6 board, indexed 0–17 from top-left to bottom-right. Most action slots are geometrically impossible for any given board state and will always be masked out at runtime.

## Rewards

| Outcome | Winner | Loser |
| --- | --- | --- |
| Win by elimination | +1 | -1 |
| Win by blocking | +1 | -1 |
| Draw (no-progress) | 0 | 0 |
| Truncation (max steps) | 0 | 0 |

## Termination

An episode terminates when:
- One player captures all of the opponent's pieces
- One player has no legal moves remaining (blocked)

An episode is truncated when:
- 20 consecutive turns pass without a capture (no-progress draw rule)
- 200 total moves are reached

## Version History

- v0: Initial release

## Usage

```python
from mycheckersenv import env

environment = env(render_mode="human")
environment.reset(seed=42)

for agent in environment.agent_iter():
    observation, reward, termination, truncation, info = environment.last()

    if termination or truncation:
        action = None
    else:
        mask = info["action_mask"]
        action = environment.action_space(agent).sample(mask)

    environment.step(action)
environment.close()
```
