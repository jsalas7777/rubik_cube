import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Cube, FACES } from './cube.js';
import { loadSolver } from './solver.js';

const canvas = document.querySelector('#scene');
const statusEl = document.querySelector('#status');
const movesEl = document.querySelector('#moves');
const solutionEl = document.querySelector('#solution');
const solveButton = document.querySelector('#solve');
const beamSelect = document.querySelector('#beam');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.set(4.5, 4, 6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 5;
controls.maxDistance = 14;

scene.add(new THREE.AmbientLight(0xffffff, 1.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 2);
keyLight.position.set(5, 8, 6);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.8);
fillLight.position.set(-6, -4, -5);
scene.add(fillLight);

const cube = new Cube(scene);

let wasSolved = true;
let solving = false;
// Replaces the usual "Solved in N moves!" when the solver's solution finishes playing.
let solverMessage = null;

function setStatus(state, text) {
  statusEl.dataset.state = state;
  statusEl.textContent = text;
}

cube.onChange = () => {
  const count = cube.history.length;
  movesEl.textContent = `${count} move${count === 1 ? '' : 's'}`;
  if (cube.busy) return;

  const solved = cube.isSolved();
  if (solved && !wasSolved) setStatus('solved', solverMessage ?? (count > 0 ? `Solved in ${count} moves!` : 'Solved'));
  else setStatus(solved ? 'solved' : 'unsolved', solved ? 'Solved' : 'Scrambled');
  solverMessage = null;
  wasSolved = solved;
};
cube.onChange();

const waitForIdle = () =>
  new Promise((resolve) => {
    const check = () => (cube.busy ? requestAnimationFrame(check) : resolve());
    check();
  });

async function solve() {
  solving = true;
  solveButton.disabled = true;
  setStatus('thinking', 'Loading model…');
  try {
    const solver = await loadSolver();
    await waitForIdle();
    const started = performance.now();
    const solution = await solver.solve(solver.readState(cube), {
      beamWidth: Number(beamSelect.value),
      onProgress: (depth) => setStatus('thinking', `Thinking… depth ${depth}`),
    });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);

    if (solution === null) {
      setStatus('unsolved', 'No solution found');
      solutionEl.textContent = `No solution found in ${seconds}s. Try a wider search.`;
    } else if (solution.length === 0) {
      setStatus('solved', 'Solved');
    } else {
      solutionEl.textContent = `AI solution, ${solution.length} moves in ${seconds}s: ${solution.join(' ')}`;
      solverMessage = `AI solved it in ${solution.length} moves!`;
      solution.forEach((move) => cube.move(move, { record: false, duration: 260 }));
      cube.history = [];
      await waitForIdle();
    }
  } catch (error) {
    console.error(error);
    setStatus('unsolved', 'Solver error');
    solutionEl.textContent = error.message;
  } finally {
    solving = false;
    solveButton.disabled = false;
  }
}

// Every input except camera orbiting is ignored while the solver thinks or plays its solution.
const whenIdle = (action) => (...args) => {
  if (!solving) action(...args);
};
const userMove = whenIdle((move) => cube.move(move));

document.querySelectorAll('[data-move]').forEach((button) => {
  button.addEventListener('click', () => userMove(button.dataset.move));
});
document.querySelector('#scramble').addEventListener('click', whenIdle(() => {
  solutionEl.textContent = '';
  cube.scramble();
}));
document.querySelector('#undo').addEventListener('click', whenIdle(() => cube.undo()));
document.querySelector('#reset').addEventListener('click', whenIdle(() => {
  wasSolved = true;
  solutionEl.textContent = '';
  cube.reset();
}));
solveButton.addEventListener('click', whenIdle(solve));

window.addEventListener('keydown', (event) => {
  if (event.metaKey || event.altKey || event.target === beamSelect) return;
  if (event.ctrlKey) {
    if (event.key.toLowerCase() === 'z') {
      event.preventDefault();
      whenIdle(() => cube.undo())();
    }
    return;
  }
  const face = event.key.toUpperCase();
  if (face in FACES) userMove(event.shiftKey ? `${face}'` : face);
});

function resize() {
  const { clientWidth: width, clientHeight: height } = canvas;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

renderer.setAnimationLoop((now) => {
  cube.update(now);
  controls.update();
  renderer.render(scene, camera);
});
