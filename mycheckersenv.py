"""
6x6 Checkers environment following the PettingZoo AEC API.

player_0 starts at the bottom (rows 4-5) and moves up.
player_1 starts at the top (rows 0-1) and moves down.
Only dark squares are used — 18 playable squares total.
"""


import functools
from copy import deepcopy

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
try:
    from pettingzoo.utils import AgentSelector          # >= 1.25.0
except ImportError:
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector  # 1.24.x



BOARD_ROWS = 6
BOARD_COLS = 6
NUM_PLAYABLE_SQUARES = 18

# board cell values
EMPTY = 0
P0_MAN = 1
P0_KING = 2
P1_MAN = 3
P1_KING = 4

# actions: (from_square * 4 + direction), directions are up-left/up-right/down-left/down-right
# 0-71: simple moves, 72-143: jumps
NUM_SIMPLE_ACTIONS = NUM_PLAYABLE_SQUARES * 4
NUM_JUMP_ACTIONS = NUM_PLAYABLE_SQUARES * 4
NUM_ACTIONS = NUM_SIMPLE_ACTIONS + NUM_JUMP_ACTIONS

OBS_SIZE = BOARD_ROWS * BOARD_COLS

MAX_STEPS = 200
CAPTURE_REWARD = 0.0
NO_PROGRESS_LIMIT = 20

CHECK_LEGALITY = True


#map between playable square indices (0-17) and (row, col) coordinates.
def _build_square_maps():
    sq_to_rc = {}
    rc_to_sq = {}
    idx = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (r + c) % 2 == 1:
                sq_to_rc[idx] = (r, c)
                rc_to_sq[(r, c)] = idx
                idx += 1
    return sq_to_rc, rc_to_sq


SQ_TO_RC, RC_TO_SQ = _build_square_maps()


def _build_action_maps():
    """
    Pre-compute the fixed action index ↔ (from_sq, to_sq / landing_sq) tables.

    Simple moves (indices 0–71):
        action = from_sq * 4 + direction_idx
        direction_idx: 0=up-left, 1=up-right, 2=down-left, 3=down-right
        to_sq = one diagonal step away (or -1 if off-board / not dark)

    Jump moves (indices 72–143):
        action = 72 + from_sq * 4 + direction_idx
        over_sq  = the piece being jumped (middle square)
        land_sq  = landing square two diagonal steps away
    """
    directions = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]

    simple_map = {}
    jump_map = {}

    for sq in range(NUM_PLAYABLE_SQUARES):
        r, c = SQ_TO_RC[sq]
        for d_idx, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            to_sq = RC_TO_SQ.get((nr, nc), -1)
            simple_map[sq * 4 + d_idx] = (sq, to_sq)

            mr, mc = r + dr, c + dc
            lr, lc = r + 2 * dr, c + 2 * dc
            over_sq = RC_TO_SQ.get((mr, mc), -1)
            land_sq = RC_TO_SQ.get((lr, lc), -1)
            jump_map[NUM_SIMPLE_ACTIONS + sq * 4 + d_idx] = (sq, over_sq, land_sq)

    return simple_map, jump_map


SIMPLE_MAP, JUMP_MAP = _build_action_maps()


def env(render_mode=None):
    """Wrap raw_env with standard PettingZoo wrappers."""
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    environment = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class raw_env(AECEnv):
    """
    6x6 Checkers — PettingZoo AEC environment.
 
    Observation: flat 36-cell board, player-relative (my pieces always at bottom).
    Actions: Discrete (144), masked at runtime via infos["action_mask"].
    Rewards: +1/-1 on win/loss, 0 on draw/truncation.
    Termination: opponent eliminated or has no legal moves.
    Truncation: MAX_STEPS reached, or NO_PROGRESS_LIMIT consecutive non-capture turns.
    """
 
    metadata = {"render_modes": ["human", "ansi"], "name": "checkers_v0"}
 
    def __init__(self, render_mode=None):
        self.possible_agents    = ["player_0", "player_1"]
        self.agent_name_mapping = {"player_0": 0, "player_1": 1}
        self.render_mode        = render_mode
        self.board              = None
        self._mid_jump_agent    = None
        self._mid_jump_sq       = None
        self._captures_this_turn = 0
        self._no_progress_count  = 0
 
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=4, shape=(OBS_SIZE,), dtype=np.int8)
 
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(NUM_ACTIONS)
 
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
 
        self.agents              = self.possible_agents[:]
        self.rewards             = {a: 0     for a in self.agents}
        self._cumulative_rewards = {a: 0     for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {}    for a in self.agents}
 
        self.board               = self._initial_board()
        self.num_moves           = 0
        self._mid_jump_agent     = None
        self._mid_jump_sq        = None
        self._captures_this_turn = 0
        self._no_progress_count  = 0
 
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
 
        for a in self.agents:
            self.infos[a] = {"action_mask": self._compute_action_mask(a)}
 
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
 
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.rewards = {a: 0 for a in self.agents}
 
        if CHECK_LEGALITY:
            mask = self.infos[agent]["action_mask"]
            if not mask[action]:
                raise ValueError(
                    f"{agent} submitted illegal action {action}. "
                    f"Legal: {np.where(mask)[0].tolist()}"
                )
 
        landing_sq = self._apply_action(agent, action)
 
        if landing_sq is not None:
            self._captures_this_turn += 1
 
        # check for multi-jump continuation
        if landing_sq is not None:
            lr, lc = SQ_TO_RC[landing_sq]
            just_promoted = (
                (agent == "player_0" and self.board[lr, lc] == P0_KING and lr == 0) or
                (agent == "player_1" and self.board[lr, lc] == P1_KING and lr == 5)
            )
            if not just_promoted:
                cont_mask = self._compute_continuation_mask(agent, landing_sq)
                if np.any(cont_mask):
                    self._mid_jump_agent = agent
                    self._mid_jump_sq    = landing_sq
                    self.infos[agent]    = {"action_mask": cont_mask}
                    self._accumulate_rewards()
                    if self.render_mode == "human":
                        self.render()
                    return
 
            self._mid_jump_agent = None
            self._mid_jump_sq    = None
 
        self.num_moves += 1
        opponent = self._opponent(agent)
 
        if self._captures_this_turn > 0:
            self.rewards[agent]    =  CAPTURE_REWARD * self._captures_this_turn
            self.rewards[opponent] = -CAPTURE_REWARD * self._captures_this_turn
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1
        self._captures_this_turn = 0
 
        game_over, winner = self._check_game_over(agent, opponent)
        if game_over:
            if winner == agent:
                self.rewards[agent] =  1; self.rewards[opponent] = -1
            elif winner == opponent:
                self.rewards[agent] = -1; self.rewards[opponent] =  1
            else:
                self.rewards[agent] =  0; self.rewards[opponent] =  0
            self.terminations = {a: True for a in self.agents}
 
        if self._no_progress_count >= NO_PROGRESS_LIMIT:
            self.rewards[agent] = self.rewards[opponent] = 0
            self.truncations = {a: True for a in self.agents}
 
        if self.num_moves >= MAX_STEPS:
            self.truncations = {a: True for a in self.agents}
 
        for a in self.agents:
            self.infos[a] = {"action_mask": self._compute_action_mask(a)}
 
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
 
        if self.render_mode == "human":
            self.render()
 
    def observe(self, agent):
        """
        Player-relative observation: each agent always sees its own pieces at rows 4-5 and
        opponent pieces at rows 0-1 (values 3/4).
        """
        if agent == "player_0":
            return self.board.flatten().astype(np.int8)
 
        flipped  = np.flipud(self.board).copy()
        remapped = np.zeros_like(flipped)
        remapped[flipped == P1_MAN]  = P0_MAN
        remapped[flipped == P1_KING] = P0_KING
        remapped[flipped == P0_MAN]  = P1_MAN
        remapped[flipped == P0_KING] = P1_KING
        return remapped.flatten().astype(np.int8)
 
    def render(self):
        if self.render_mode is None:
            return
        symbols = {EMPTY: ".", P0_MAN: "o", P0_KING: "O", P1_MAN: "x", P1_KING: "X"}
        print("\n  0 1 2 3 4 5")
        for r in range(BOARD_ROWS):
            print(f"{r} " + " ".join(symbols[self.board[r, c]] for c in range(BOARD_COLS)))
        print(f"  Move {self.num_moves} | Next: {self.agent_selection}\n")
 
    def close(self):
        pass
 
    def _initial_board(self):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        for r in range(2):
            for c in range(BOARD_COLS):
                if (r + c) % 2 == 1:
                    board[r, c] = P1_MAN
        for r in range(4, 6):
            for c in range(BOARD_COLS):
                if (r + c) % 2 == 1:
                    board[r, c] = P0_MAN
        return board
 
    def _compute_action_mask(self, agent):
        """
        Returns legal actions for agent.
        Jumps are mandatory — if any exist, only jumps are returned.
        During a multi-jump chain, only continuations from the current square are returned.
        """
        if self._mid_jump_agent == agent:
            return self._compute_continuation_mask(agent, self._mid_jump_sq)
 
        mask      = np.zeros(NUM_ACTIONS, dtype=np.int8)
        my_men    = P0_MAN  if agent == "player_0" else P1_MAN
        my_kings  = P0_KING if agent == "player_0" else P1_KING
        opp_men   = P1_MAN  if agent == "player_0" else P0_MAN
        opp_kings = P1_KING if agent == "player_0" else P0_KING
        fwd_rows  = [-1] if agent == "player_0" else [+1]
        DIRS = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
 
        jump_indices = []
        for sq in range(NUM_PLAYABLE_SQUARES):
            r, c = SQ_TO_RC[sq]
            cell = self.board[r, c]
            if cell not in (my_men, my_kings):
                continue
            move_dirs = fwd_rows if cell == my_men else [-1, +1]
            for d_idx, (dr, dc) in enumerate(DIRS):
                if dr not in move_dirs:
                    continue
                action_idx = NUM_SIMPLE_ACTIONS + sq * 4 + d_idx
                _, over_sq, land_sq = JUMP_MAP[action_idx]
                if over_sq == -1 or land_sq == -1:
                    continue
                or_, oc_ = SQ_TO_RC[over_sq]
                lr_, lc_ = SQ_TO_RC[land_sq]
                if self.board[or_, oc_] not in (opp_men, opp_kings):
                    continue
                if self.board[lr_, lc_] != EMPTY:
                    continue
                jump_indices.append(action_idx)
 
        if jump_indices:
            for idx in jump_indices:
                mask[idx] = 1
            return mask
 
        for sq in range(NUM_PLAYABLE_SQUARES):
            r, c = SQ_TO_RC[sq]
            cell = self.board[r, c]
            if cell not in (my_men, my_kings):
                continue
            move_dirs = fwd_rows if cell == my_men else [-1, +1]
            for d_idx, (dr, dc) in enumerate(DIRS):
                if dr not in move_dirs:
                    continue
                action_idx = sq * 4 + d_idx
                _, to_sq = SIMPLE_MAP[action_idx]
                if to_sq == -1:
                    continue
                tr_, tc_ = SQ_TO_RC[to_sq]
                if self.board[tr_, tc_] != EMPTY:
                    continue
                mask[action_idx] = 1
 
        return mask
 
    def _compute_continuation_mask(self, agent, from_sq):
        """Returns jump actions from from_sq during a multi-jump chain."""
        mask      = np.zeros(NUM_ACTIONS, dtype=np.int8)
        r, c      = SQ_TO_RC[from_sq]
        cell      = self.board[r, c]
        my_men    = P0_MAN  if agent == "player_0" else P1_MAN
        my_kings  = P0_KING if agent == "player_0" else P1_KING
        opp_men   = P1_MAN  if agent == "player_0" else P0_MAN
        opp_kings = P1_KING if agent == "player_0" else P0_KING
        fwd_rows  = [-1] if agent == "player_0" else [+1]
        DIRS = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
 
        if cell not in (my_men, my_kings):
            return mask
 
        move_dirs = fwd_rows if cell == my_men else [-1, +1]
        for d_idx, (dr, dc) in enumerate(DIRS):
            if dr not in move_dirs:
                continue
            action_idx = NUM_SIMPLE_ACTIONS + from_sq * 4 + d_idx
            _, over_sq, land_sq = JUMP_MAP[action_idx]
            if over_sq == -1 or land_sq == -1:
                continue
            or_, oc_ = SQ_TO_RC[over_sq]
            lr_, lc_ = SQ_TO_RC[land_sq]
            if self.board[or_, oc_] not in (opp_men, opp_kings):
                continue
            if self.board[lr_, lc_] != EMPTY:
                continue
            mask[action_idx] = 1
 
        return mask
 
    def _apply_action(self, agent, action):
        """
        Applies action to board. Returns landing square for jumps, None for simple moves.
        King promotion is applied immediately after each hop.
        """
        if action < NUM_SIMPLE_ACTIONS:
            from_sq, to_sq = SIMPLE_MAP[action]
            if to_sq == -1:
                raise ValueError(f"Illegal simple action {action}")
            fr, fc = SQ_TO_RC[from_sq]
            tr, tc = SQ_TO_RC[to_sq]
            self.board[tr, tc] = self.board[fr, fc]
            self.board[fr, fc] = EMPTY
            self._promote_kings()
            return None
        else:
            from_sq, over_sq, land_sq = JUMP_MAP[action]
            if over_sq == -1 or land_sq == -1:
                raise ValueError(f"Illegal jump action {action}")
            fr, fc   = SQ_TO_RC[from_sq]
            or_, oc_ = SQ_TO_RC[over_sq]
            lr, lc   = SQ_TO_RC[land_sq]
            self.board[lr, lc]   = self.board[fr, fc]
            self.board[fr, fc]   = EMPTY
            self.board[or_, oc_] = EMPTY
            self._promote_kings()
            return land_sq
 
    def _promote_kings(self):
        for c in range(BOARD_COLS):
            if self.board[0, c] == P0_MAN:
                self.board[0, c] = P0_KING
            if self.board[5, c] == P1_MAN:
                self.board[5, c] = P1_KING
 
    def _check_game_over(self, last_agent, opponent):
        """Returns (game_over, winner). winner is None for a draw."""
        opp_men   = P1_MAN  if last_agent == "player_0" else P0_MAN
        opp_kings = P1_KING if last_agent == "player_0" else P0_KING
        if not np.any(np.isin(self.board, [opp_men, opp_kings])):
            return True, last_agent
        if not np.any(self._compute_action_mask(opponent)):
            return True, last_agent
        return False, None
 
    def _opponent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"
 