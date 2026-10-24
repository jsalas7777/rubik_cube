# Rubik's Cube

An interactive 3D Rubik's cube that runs in the browser, built with [Three.js](https://threejs.org/). No build step and no dependencies to install. Three.js is loaded from a CDN.

## Running it

The page uses ES modules, so it has to be served over HTTP. Opening `index.html` directly from disk won't work.

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Any static server works, for example `npx serve`.

## Controls

| Action | Mouse / touch | Keyboard |
| --- | --- | --- |
| Look around | Drag the cube, scroll to zoom | |
| Turn a face clockwise | `U` `D` `L` `R` `F` `B` buttons | `U` `D` `L` `R` `F` `B` |
| Turn counter-clockwise | `U'` … `B'` buttons | `Shift` + face key |
| Undo | Undo button | `Ctrl` + `Z` |
| Scramble / reset | Scramble, Reset buttons | |

Moves use standard [Singmaster notation](https://en.wikipedia.org/wiki/Rubik%27s_Cube#Move_notation). They are relative to the cube's fixed orientation (white on top, green in front), not to the camera angle.

## How it works

- **`src/cube.js`** holds the cube model. The cube is made of 26 "cubies", each a rounded black box with colored sticker planes on its outer faces.
  - **A turn:** every cubie in the layer is attached to a temporary pivot group. The pivot rotates 90° with an ease-out animation, and then each cubie goes back to the cube with its position and rotation snapped to the grid. Snapping stops floating-point drift from building up over many turns.
  - **The move queue:** turns go into a queue, so fast key presses play one after another.
  - **Solved check:** the cube counts as solved when every sticker facing the same direction has the same color.
- **`src/main.js`** sets up the scene: renderer, camera, lights, orbit controls, buttons and keyboard input.
- **`index.html` / `style.css`** contain the page layout and theme, including dark mode.

## AI solver

**Solve with AI** uses a neural network trained in [`training/`](training/). The approach follows [EfficientCube (Takano, 2023)](https://arxiv.org/abs/2106.03157).

1. **Training data.** Start from a solved cube and make random moves. Each position is labelled with the move that undoes the last turn, so no human data or existing solver is needed ([`data.py`](training/data.py)).
2. **Model.** The network reads the 54 sticker colors, one-hot encoded as 324 inputs. It is a 512-wide input layer plus 3 residual blocks, about 1.8M parameters, and it predicts which of the 12 moves heads back toward solved ([`model.py`](training/model.py), [`train.py`](training/train.py)). Every batch is freshly generated, so it never sees the same data twice.
3. **Search.** Beam search keeps the most likely move sequences according to the network, extending them one move at a time until one reaches solved ([`search.py`](training/search.py) in Python, [`src/solver.js`](src/solver.js) in the browser). A wider search finds shorter solutions but takes longer.

The page runs the exported model in the browser with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/). No server is needed.

### Training your own model

```bash
cd training
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python data.py                  # sample dataset + fixed test scrambles in data/
.venv/bin/python train.py --steps 20000   # ~90 min on a 4-core CPU; --resume continues a run
.venv/bin/python search.py --beam 1000    # solve rate and solution length on the test scrambles
.venv/bin/python export.py                # writes model/solver.onnx + model/solver.json for the page
```

[`cube_env.py`](training/cube_env.py) is a NumPy cube simulator that uses the same geometry and move directions as [`src/cube.js`](src/cube.js). `export.py` writes the sticker order and move tables into `solver.json`, so the page and the model always agree on the cube layout.

## Ideas for next steps

- Drag across a face to turn it (raycast the sticker you clicked, then pick the axis from the drag direction)
- Slice moves (`M`, `E`, `S`) and whole-cube rotations (`x`, `y`, `z`)
- A solve timer and best-time tracking
- A bigger model trained on a GPU, for shorter solutions with a narrower search
