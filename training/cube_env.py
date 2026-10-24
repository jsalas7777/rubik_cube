"""NumPy Rubik's cube simulator.

Uses the same geometry and move conventions as src/cube.js: y is up, F faces +z, and a
clockwise turn is a negative rotation about the face's outward normal. A state is an array
of 54 color indices, one per sticker slot, where a color index is the face it belongs to
when solved.
"""

import itertools

import numpy as np

# Face -> (axis, layer), mirroring FACES in src/cube.js.
FACES = {
    "U": ("y", 1),
    "D": ("y", -1),
    "L": ("x", -1),
    "R": ("x", 1),
    "F": ("z", 1),
    "B": ("z", -1),
}
FACE_ORDER = list(FACES)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

MOVES = [face + suffix for face in FACE_ORDER for suffix in ("", "'")]
NUM_MOVES = len(MOVES)


def invert_move(move):
    return move[:-1] if move.endswith("'") else move + "'"


INVERSE = np.array([MOVES.index(invert_move(m)) for m in MOVES])


def _rotation(axis, quarter_turns):
    c = round(np.cos(quarter_turns * np.pi / 2))
    s = round(np.sin(quarter_turns * np.pi / 2))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _sticker_slots():
    slots = []
    for face in FACE_ORDER:
        axis, layer = FACES[face]
        normal = [0, 0, 0]
        normal[AXIS_INDEX[axis]] = layer
        for position in itertools.product((-1, 0, 1), repeat=3):
            if position[AXIS_INDEX[axis]] == layer:
                slots.append((position, tuple(normal)))
    return slots


# Each slot is (cubie position, outward normal); the web page reads its state in this order.
STICKERS = _sticker_slots()
SOLVED = np.repeat(np.arange(len(FACE_ORDER)), 9).astype(np.uint8)


def _move_permutation(move):
    axis, layer = FACES[move[0]]
    direction = -1 if move.endswith("'") else 1
    rotation = _rotation(axis, -layer * direction)
    slot_index = {slot: i for i, slot in enumerate(STICKERS)}

    # After the move, slot k holds the sticker that was in slot perm[k].
    perm = np.arange(len(STICKERS))
    for j, (position, normal) in enumerate(STICKERS):
        if position[AXIS_INDEX[axis]] != layer:
            continue
        k = slot_index[(tuple(rotation @ position), tuple(rotation @ normal))]
        perm[k] = j
    return perm


PERMS = np.stack([_move_permutation(m) for m in MOVES])


def apply_moves(states, moves):
    """Apply one move per state. states: (N, 54) uint8, moves: (N,) move indices."""
    return np.take_along_axis(states, PERMS[moves], axis=1)


def apply_sequence(state, sequence):
    """Apply a sequence of moves (names or indices) to a single state."""
    for move in sequence:
        index = MOVES.index(move) if isinstance(move, str) else move
        state = state[PERMS[index]]
    return state


def is_solved(states):
    return np.all(states == SOLVED, axis=-1)
