// The tiny long-division machine of cpu_first/bridge (grid.py) written as a handwired program: one tick of the grid,
// built with handwired's Circuit and cells.ts, run by a host loop that writes each tick's next states back.
//
// The machine (TINYROUTING's division core with H1's commit-gate chain): base B, moduli N of w digits, dividends
// v < N * B^k, answer v mod N, digits lowest first. A register of P = w + 1 places starts at v's top w digits; each of
// the k steps shifts it up one place and brings the next digit of v in, then runs B - 1 links: subtract N with a
// ripple of borrows, and keep the result only if the subtraction did not borrow.
//
// As a grid (the bridge's layout, so the two bridges compare cell for cell): every table application is one cell, a
// row of P + 1 cells holds one register at columns -1 .. P - 1, and the rows are N, R0, then per step SHIFT and per
// link SUB, SEL. Each cell holds a one-hot digit (B exact bits) and a one-hot borrow (2 exact bits: none, borrow).
//
// One tick computes every scratch cell's next state from the current states of the cells it reads (its ports A, B, C):
//   SHIFT  digit = A's digit                              a copy
//   SUB    (digit, borrow) = table[A's digit][B's digit][C's borrow]: a - b - c, one place of a link's ripple
//   SEL    digit = C's borrow ? A's digit : B's digit    the commit gate, a select
// Input cells (N's and v's digits, constant zeros) are input cells: the host writes them, the program keeps them.
// After depth = k (1 + (B - 1)(P + 1)) ticks every cell holds the machine's value, from any start.
//
// What each part becomes in the compiled network (affine plus ReGLU, nothing else):
//   the table   a sum over the cell's joint one-hot: products a_i * b_j * c_k (two ReGLU multiplies), then one affine
//               row per output bit whose coefficients are the table, 0 or 1 ("a value that depends on a one-hot is one
//               sum over the one-hot", patterns.md)
//   the gate    emitSelect: C's borrow * A's digit + (1 - C's borrow) * B's digit, ReGLU products
//   the copy    a one-term affine kept as its own operation (roundAffine), so the copy is a weight and not a host alias
//   the wiring  which cells' wires each operation reads: the sparsity pattern of the layers' weights
//   the loop    the host's tick loop: the same compiled program evaluated depth times (tied weights)
import {Circuit, type Wire, emitSelect} from '../../../../handwired/src/runtime/numeric-ui-core.ts'
import {createSourceTable, inputBit, list, oneHot, stateBit} from '../../../../handwired/src/runtime/cells.ts'

export const KINDS = ['Z', 'NIN', 'VIN', 'SHIFT', 'SUB', 'SEL'] as const
export const [Z, NIN, VIN, SHIFT, SUB, SEL] = [0, 1, 2, 3, 4, 5]

export type Layout = ReturnType<typeof layout>

// The grid for one size, as grid.Layout builds it: kinds, coordinates, where input cells take their digit (an index
// into N's w digits or v's w + k digits, -1 for a constant zero), and the three ports of every scratch cell.
export function layout(B: number, w: number, k: number, links = B - 1) {
  const P = w + 1
  const C = P + 1
  const rows: [name: string, step: number, link: number][] = [['N', -1, 0], ['R0', -1, 0]]
  for (let t = 0; t < k; t++) {
    rows.push(['SHIFT', t, 0])
    for (let s = 1; s <= links; s++) rows.push(['SUB', t, s], ['SEL', t, s])
  }
  const Rn = rows.length
  const L = Rn * C
  const cell = (r: number, c: number) => r * C + c + 1
  const kind = Array<number>(L).fill(Z)
  const src = Array<number>(L).fill(-1)
  const row = Array.from({length: L}, (_, p) => Math.floor(p / C))
  const col = Array.from({length: L}, (_, p) => (p % C) - 1)
  for (let c = 0; c < w; c++) {
    kind[cell(0, c)] = NIN
    src[cell(0, c)] = c
    kind[cell(1, c)] = VIN
    src[cell(1, c)] = k + c
  }
  kind[cell(1, -1)] = VIN
  src[cell(1, -1)] = k - 1
  rows.forEach(([name, t, s], r) => {
    if (name === 'SHIFT' || name === 'SUB' || name === 'SEL') {
      for (let c = 0; c < P; c++) kind[cell(r, c)] = KINDS.indexOf(name as typeof KINDS[number])
    }
    if (name === 'SEL' && s === links && t < k - 1) {
      kind[cell(r, -1)] = VIN
      src[cell(r, -1)] = k - 2 - t
    }
  })
  // Each scratch cell's ports, relative to the cell at (r, c); -1 for a port its kind does not read.
  const ports = [0, 1, 2].map(() => Array<number>(L).fill(-1))
  for (let p = 0; p < L; p++) {
    const r = row[p]!
    const c = col[p]!
    if (kind[p] === SHIFT) {
      ports[0]![p] = cell(r - 1, c - 1)
    } else if (kind[p] === SUB) {
      ports[0]![p] = cell(r - 1, c)
      ports[1]![p] = cell(0, c)
      ports[2]![p] = cell(r, c - 1)
    } else if (kind[p] === SEL) {
      ports[0]![p] = cell(r - 2, c)
      ports[1]![p] = cell(r - 1, c)
      ports[2]![p] = cell(r - 1, P - 1)
    }
  }
  const answer = Array.from({length: w}, (_, c) => cell(Rn - 1, c))
  return {B, w, k, links, P, C, Rn, L, rows, kind, row, col, src, ports, answer, depth: k * (1 + links * (P + 1))}
}

// What an operation is, for the PyTorch side: the cell it belongs to and its role in that cell. handwired shares an
// operation between cells when two build the same expression, so a wire can carry several tags.
export type Tag = {cell: number, role: string, i?: number, j?: number, k?: number, m?: number}

// The tick program for one size: the circuit, the next value of every source cell, and each grid cell's wires.
export function buildTick(lay: Layout) {
  const {B, L} = lay
  const table = createSourceTable()
  const isScratch = (p: number) => lay.kind[p] === SHIFT || lay.kind[p] === SUB || lay.kind[p] === SEL
  // A blank cell (and every scratch cell's start) is digit 0 with no borrow. Input cells are input bits the host
  // writes; scratch cells are state bits the program computes. The one-hot groups are checked by the validator only:
  // declared to the circuit they would let the compiler turn masked sums into gathers, which are not affine.
  const bit = (p: number, initial: number) => isScratch(p) ? stateBit(initial) : inputBit(initial)
  const grid = table.declare(list(L, p => ({
    digit: oneHot(list(B, i => bit(p, Number(i === 0))), {declareToCircuit: false}),
    borrow: oneHot([bit(p, 1), bit(p, 0)], {declareToCircuit: false}),
  })), 'grid')
  const circuit = table.createCircuit()
  const tags = new Map<Wire, Tag[]>()
  const tag = (wire: Wire, t: Tag) => {
    const list = tags.get(wire) ?? []
    list.push(t)
    tags.set(wire, list)
    return wire
  }
  const one = tag(circuit.constant(1), {cell: -1, role: 'one'})
  const zero = tag(circuit.constant(0), {cell: -1, role: 'zero'})
  for (let p = 0; p < L; p++) {
    if (!isScratch(p)) continue
    const [A, Bp, C] = lay.ports.map(port => port[p]! >= 0 ? grid[port[p]!]! : undefined)
    const noBorrow = [one, zero]
    if (lay.kind[p] === SHIFT) {
      // JS: digit = A.digit, borrow = none
      const digit = A!.digit.map((wire, i) => tag(circuit.roundAffine([[wire, 1]]), {cell: p, role: 'copy', i}))
      table.next(grid[p]!, {digit, borrow: noBorrow})
    } else if (lay.kind[p] === SUB) {
      // JS: x = a - b - c; digit = (x + B) % B, borrow = x < 0, as one sum over the joint one-hot of (a, b, c)
      const joint: {wire: Wire, i: number, j: number, k: number}[] = []
      for (let i = 0; i < B; i++) {
        for (let j = 0; j < B; j++) {
          const ab = tag(circuit.multiply(A!.digit[i]!, Bp!.digit[j]!), {cell: p, role: 'ab', i, j})
          for (let k = 0; k < 2; k++) {
            joint.push({wire: tag(circuit.multiply(ab, C!.borrow[k]!), {cell: p, role: 'abc', i, j, k}), i, j, k})
          }
        }
      }
      const out = (m: number) => tag(circuit.affine(joint
        .filter(({i, j, k}) => m < B ? (i - j - k + 2 * B) % B === m : Number(i - j - k < 0) === m - B)
        .map(({wire}) => [wire, 1] as const)), {cell: p, role: 'out', m})
      table.next(grid[p]!, {digit: list(B, out), borrow: [out(B), out(B + 1)]})
    } else {
      // JS: digit = C.borrow ? A.digit : B.digit, borrow = none
      const digit = list(B, i => tag(emitSelect(circuit, C!.borrow[1]!, Bp!.digit[i]!, A!.digit[i]!), {cell: p, role: 'select', i}))
      table.next(grid[p]!, {digit, borrow: noBorrow})
    }
  }
  const built = table.finish()
  return {...built, grid, tags, isScratch}
}

// Every question of a size, as grid.Data enumerates them: N with w digits (top digit nonzero, N >= 2), v < N B^k.
export function questions(B: number, w: number, k: number) {
  const out: {N: number, v: number}[] = []
  for (let N = Math.max(2, B ** (w - 1)); N < B ** w; N++) {
    for (let v = 0; v < N * B ** k; v++) out.push({N, v})
  }
  return out
}

export const digitsOf = (x: number, B: number, n: number) => list(n, i => Math.floor(x / B ** i) % B)

// The host: write one question's input digits into a source table whose scratch cells hold the blank start.
export function writeQuestion(lay: Layout, built: ReturnType<typeof buildTick>, source: Float32Array, N: number, v: number) {
  const Nd = digitsOf(N, lay.B, lay.w)
  const vd = digitsOf(v, lay.B, lay.w + lay.k)
  for (let p = 0; p < lay.L; p++) {
    if ((lay.kind[p] !== NIN && lay.kind[p] !== VIN) || lay.src[p]! < 0) continue
    const d = lay.kind[p] === NIN ? Nd[lay.src[p]!]! : vd[lay.src[p]!]!
    built.grid[p]!.digit.forEach((wire, i) => { source[wire] = Number(i === d) })
  }
}

export function readAnswer(lay: Layout, built: ReturnType<typeof buildTick>, source: Float32Array) {
  let y = 0
  lay.answer.forEach((p, place) => {
    const bits = built.grid[p]!.digit.map(wire => source[wire]!)
    y += bits.indexOf(1) * lay.B ** place
  })
  return y
}

export {Circuit}
