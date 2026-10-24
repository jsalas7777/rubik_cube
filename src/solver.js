// Runs the trained policy network in the browser and solves with beam search,
// mirroring training/search.py. Files come from training/export.py.
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MODEL_URL = 'model/solver.onnx';
const LAYOUT_URL = 'model/solver.json';

let solverPromise = null;

export function loadSolver() {
  solverPromise ??= createSolver().catch((error) => {
    solverPromise = null;
    throw error;
  });
  return solverPromise;
}

async function createSolver() {
  const ort = await import(`${ORT_URL}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_URL;
  const layoutResponse = await fetch(LAYOUT_URL);
  if (!layoutResponse.ok) throw new Error('Model not found. Run training/export.py first.');
  const [layout, session] = await Promise.all([
    layoutResponse.json(),
    ort.InferenceSession.create(MODEL_URL),
  ]);
  return new Solver(ort, session, layout);
}

const slotIndexOf = (layout) => new Map(layout.stickers.map((slot, i) => [slot.join(','), i]));

// Converts the cube's stickers into the 54-color state the network was trained on.
export function readState(cube, layout, slotIndex = slotIndexOf(layout)) {
  const state = new Uint8Array(layout.stickers.length);
  for (const { position, normal, face } of cube.stickers()) {
    state[slotIndex.get([...position, ...normal].join(','))] = layout.faces.indexOf(face);
  }
  return state;
}

class Solver {
  constructor(ort, session, layout) {
    this.ort = ort;
    this.session = session;
    this.layout = layout;
    this.slotIndex = slotIndexOf(layout);
    this.solvedState = Uint8Array.from(layout.stickers, (_, i) => Math.floor(i / 9));
  }

  readState(cube) {
    return readState(cube, this.layout, this.slotIndex);
  }

  isSolved(state) {
    return state.every((color, i) => color === this.solvedState[i]);
  }

  async predict(states) {
    const colors = this.layout.faces.length;
    const width = states[0].length * colors;
    const input = new Float32Array(states.length * width);
    states.forEach((state, i) => {
      state.forEach((color, s) => {
        input[i * width + s * colors + color] = 1;
      });
    });
    const feeds = { [this.session.inputNames[0]]: new this.ort.Tensor('float32', input, [states.length, width]) };
    const output = await this.session.run(feeds);
    return output[this.session.outputNames[0]].data;
  }

  // Returns the solving move list, or null if none is found within maxDepth moves.
  async solve(state, { beamWidth = 1000, maxDepth = 60, onProgress } = {}) {
    if (this.isSolved(state)) return [];
    const { moves, inverse, perms } = this.layout;
    let beam = [{ state, path: [], score: 0 }];

    for (let depth = 1; depth <= maxDepth; depth++) {
      const logProbs = await this.predict(beam.map((node) => node.state));
      const candidates = [];
      beam.forEach((node, i) => {
        const last = node.path.at(-1);
        for (let m = 0; m < moves.length; m++) {
          // Never undo the move that was just made.
          if (last !== undefined && m === inverse[last]) continue;
          candidates.push({ node, move: m, score: node.score + logProbs[i * moves.length + m] });
        }
      });
      candidates.sort((a, b) => b.score - a.score);

      const next = [];
      const seen = new Set();
      for (const { node, move, score } of candidates) {
        if (next.length === beamWidth) break;
        const perm = perms[move];
        const child = Uint8Array.from(perm, (from) => node.state[from]);
        // Different paths often reach the same state; keep only the best-scoring copy.
        const key = child.join(',');
        if (seen.has(key)) continue;
        seen.add(key);

        const path = [...node.path, move];
        if (this.isSolved(child)) return path.map((m) => moves[m]);
        next.push({ state: child, path, score });
      }
      beam = next;
      onProgress?.(depth);
      // Yield so the page can repaint between depths.
      await new Promise((resolve) => setTimeout(resolve));
    }
    return null;
  }
}
