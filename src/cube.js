import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

// Standard color scheme: white on top, green in front.
const COLORS = {
  U: 0xffffff,
  D: 0xffd500,
  F: 0x009b48,
  B: 0x0046ad,
  R: 0xb71234,
  L: 0xff5800,
};

// Each face turns the layer at `layer` along `axis`, and its outward normal points that way.
export const FACES = {
  U: { axis: 'y', layer: 1 },
  D: { axis: 'y', layer: -1 },
  R: { axis: 'x', layer: 1 },
  L: { axis: 'x', layer: -1 },
  F: { axis: 'z', layer: 1 },
  B: { axis: 'z', layer: -1 },
};

const MOVE_PATTERN = /^([UDLRFB])(2|')?$/;
const Z_AXIS = new THREE.Vector3(0, 0, 1);

export function parseMove(notation) {
  const match = MOVE_PATTERN.exec(notation);
  if (!match) throw new Error(`Invalid move: ${notation}`);
  const [, face, modifier] = match;
  const { axis, layer } = FACES[face];
  const turns = modifier === '2' ? 2 : 1;
  const direction = modifier === "'" ? -1 : 1;
  // Clockwise as seen from outside the face is a negative rotation about its outward normal.
  return { face, axis, layer, angle: -layer * direction * turns * (Math.PI / 2) };
}

export function invertMove(notation) {
  if (notation.endsWith('2')) return notation;
  return notation.endsWith("'") ? notation.slice(0, -1) : `${notation}'`;
}

function roundedSquare(size, radius) {
  const s = size / 2;
  const shape = new THREE.Shape();
  shape.moveTo(-s + radius, -s);
  shape.lineTo(s - radius, -s);
  shape.quadraticCurveTo(s, -s, s, -s + radius);
  shape.lineTo(s, s - radius);
  shape.quadraticCurveTo(s, s, s - radius, s);
  shape.lineTo(-s + radius, s);
  shape.quadraticCurveTo(-s, s, -s, s - radius);
  shape.lineTo(-s, -s + radius);
  shape.quadraticCurveTo(-s, -s, -s + radius, -s);
  return new THREE.ShapeGeometry(shape);
}

// Snaps a cubie back onto the grid so floating-point error never accumulates across turns.
function snap(cubie) {
  cubie.position.round();
  const m = new THREE.Matrix4().makeRotationFromQuaternion(cubie.quaternion);
  for (let i = 0; i < 16; i++) m.elements[i] = Math.round(m.elements[i]);
  cubie.quaternion.setFromRotationMatrix(m);
}

export class Cube {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);

    this.bodyGeometry = new RoundedBoxGeometry(0.96, 0.96, 0.96, 3, 0.08);
    this.bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.6 });
    this.stickerGeometry = roundedSquare(0.82, 0.1);
    this.stickerMaterials = Object.fromEntries(
      Object.entries(COLORS).map(([face, color]) => [
        face,
        new THREE.MeshStandardMaterial({ color, roughness: 0.35 }),
      ]),
    );

    this.onChange = () => {};
    this.reset();
  }

  reset() {
    this.group.clear();
    this.cubies = [];
    this.queue = [];
    this.active = null;
    this.history = [];

    for (let x = -1; x <= 1; x++) {
      for (let y = -1; y <= 1; y++) {
        for (let z = -1; z <= 1; z++) {
          if (x === 0 && y === 0 && z === 0) continue;
          this.cubies.push(this.createCubie(x, y, z));
        }
      }
    }
    this.onChange();
  }

  createCubie(x, y, z) {
    const cubie = new THREE.Mesh(this.bodyGeometry, this.bodyMaterial);
    cubie.position.set(x, y, z);

    for (const [face, { axis, layer }] of Object.entries(FACES)) {
      if (cubie.position[axis] !== layer) continue;
      const normal = new THREE.Vector3();
      normal[axis] = layer;

      const sticker = new THREE.Mesh(this.stickerGeometry, this.stickerMaterials[face]);
      sticker.position.copy(normal).multiplyScalar(0.485);
      sticker.quaternion.setFromUnitVectors(Z_AXIS, normal);
      sticker.userData.face = face;
      cubie.add(sticker);
    }

    this.group.add(cubie);
    return cubie;
  }

  get busy() {
    return this.active !== null || this.queue.length > 0;
  }

  move(notation, { record = true, duration = 220 } = {}) {
    const move = parseMove(notation);
    this.queue.push({ ...move, notation, duration: notation.endsWith('2') ? duration * 1.5 : duration });
    if (record) this.history.push(notation);
  }

  undo() {
    const last = this.history.pop();
    if (last) this.move(invertMove(last), { record: false });
  }

  scramble(length = 25) {
    const faces = Object.keys(FACES);
    const modifiers = ['', "'", '2'];
    let previousAxis = null;
    for (let i = 0; i < length; i++) {
      // Skipping consecutive turns on the same axis avoids moves that cancel or merge.
      let face;
      do {
        face = faces[Math.floor(Math.random() * faces.length)];
      } while (FACES[face].axis === previousAxis);
      previousAxis = FACES[face].axis;
      this.move(face + modifiers[Math.floor(Math.random() * modifiers.length)], {
        record: false,
        duration: 70,
      });
    }
    this.history = [];
  }

  update(now) {
    if (!this.active && this.queue.length) this.start(this.queue.shift(), now);
    if (!this.active) return;

    const { pivot, axis, angle, start, duration } = this.active;
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    pivot.rotation[axis] = angle * eased;
    if (t === 1) this.finish();
  }

  start(move, now) {
    const pivot = new THREE.Group();
    this.group.add(pivot);
    for (const cubie of this.cubies) {
      if (Math.round(cubie.position[move.axis]) === move.layer) pivot.attach(cubie);
    }
    this.active = { ...move, pivot, start: now };
  }

  finish() {
    const { pivot } = this.active;
    for (const cubie of [...pivot.children]) {
      this.group.attach(cubie);
      snap(cubie);
    }
    this.group.remove(pivot);
    this.active = null;
    this.onChange();
  }

  // Every sticker as its cubie's grid position, the direction it faces, and its color.
  *stickers() {
    const normal = new THREE.Vector3();
    for (const cubie of this.cubies) {
      for (const sticker of cubie.children) {
        normal.copy(Z_AXIS).applyQuaternion(sticker.quaternion).applyQuaternion(cubie.quaternion).round();
        yield {
          position: cubie.position.toArray().map(Math.round),
          normal: normal.toArray(),
          face: sticker.userData.face,
        };
      }
    }
  }

  // Solved when every sticker facing the same direction has the same color.
  isSolved() {
    const colorsByDirection = new Map();
    for (const { normal, face } of this.stickers()) {
      const key = normal.join(',');
      if (!colorsByDirection.has(key)) colorsByDirection.set(key, face);
      else if (colorsByDirection.get(key) !== face) return false;
    }
    return true;
  }
}
