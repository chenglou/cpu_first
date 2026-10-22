// Compile the tick program at each size, check it with handwired's own evaluators, and export it for PyTorch.
//
//   bun cpu_first/handwired_bridge/js/compile.ts            every size of the attention bridge's compile table
//   bun cpu_first/handwired_bridge/js/compile.ts 3,2,1 ...  only these sizes
//
// For each size (B, w, k):
// - builds the tick (machine.ts) and compiles it with compileTensorProgram for the next values of the state cells,
//   without gather lowering, so the program holds only affine rows and ReGLU gates;
// - runs the host loop, depth evaluations writing next states back, on every question (a random sample above
//   SAMPLE questions) with the compiled tensor evaluator, and on a smaller sample with evaluateReference (the
//   definition of what a program computes), from the blank start and from random one-hot scratch states;
// - writes records/compiled/B{B}w{w}k{k}.json: the layout, every grid cell's wires, the compiled layers, the next
//   value of every source cell, the live operations with their tags, and the checks.
import {mkdirSync, writeFileSync} from 'node:fs'
import {dirname, join} from 'node:path'
import {fileURLToPath} from 'node:url'
import {type Layout, NIN, SEL, SHIFT, SUB, VIN, buildTick, layout, questions, readAnswer, writeQuestion} from './machine.ts'

const SIZES = [[3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 2, 4], [3, 3, 2], [3, 3, 3], [2, 3, 3], [2, 4, 4], [4, 2, 2], [4, 2, 3],
  [5, 2, 2], [4, 3, 2], [5, 3, 2], [3, 4, 3]]
const SAMPLE = 20_000
const REFERENCE_SAMPLE = 300
const RANDOM_START_SAMPLE = 2_000

const here = dirname(fileURLToPath(import.meta.url))
const outDir = join(here, '..', 'records', 'compiled')

// A small deterministic generator (mulberry32), so samples are the same on every run.
function random(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function sample<T>(items: T[], n: number, rand: () => number) {
  if (items.length <= n) return items
  const copy = [...items]
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(rand() * (copy.length - i))
    ;[copy[i], copy[j]] = [copy[j]!, copy[i]!]
  }
  return copy.slice(0, n)
}

const plain = (array: ArrayLike<number>) => Array.from(array)

function compileSize(B: number, w: number, k: number) {
  const lay: Layout = layout(B, w, k)
  const built = buildTick(lay)
  const {circuit, nextSource, initialSource, validate} = built
  const stateWires = nextSource.flatMap((_, wire) => {
    const p = Math.floor(wire / (B + 2))
    return built.isScratch(p) ? [wire] : []
  })
  const roots = stateWires.map(wire => nextSource[wire]!)
  const program = circuit.compileTensorProgram(roots, false)
  const evaluate = circuit.createTensorProgramEvaluator(program)
  for (const layer of program.layers) {
    if (layer.booleanRows.length || layer.indexReads.length || layer.gatherReads.length) {
      throw new Error('the tick program should hold only affine rows and ReGLU gates')
    }
  }

  // One question through the host loop: depth evaluations, each writing the state cells' next values back.
  const run = (N: number, v: number, start: Float32Array, step: (source: Float32Array) => Float32Array) => {
    const source = new Float32Array(start)
    writeQuestion(lay, built, source, N, v)
    validate(source)
    for (let tick = 0; tick < lay.depth; tick++) {
      const next = step(source)
      stateWires.forEach((wire, index) => { source[wire] = next[index]! })
    }
    validate(source)
    return readAnswer(lay, built, source)
  }
  const tensorStep = (source: Float32Array) => evaluate(source)
  const referenceStep = (source: Float32Array) => {
    const values = circuit.evaluateReference(source)
    return Float32Array.from(roots, root => values[root]!)
  }
  const rand = random(1000 * B + 100 * w + k)
  const all = questions(B, w, k)
  const checked = sample(all, SAMPLE, rand)
  let tensorRight = 0
  for (const {N, v} of checked) tensorRight += Number(run(N, v, initialSource, tensorStep) === v % N)
  let referenceRight = 0
  const referenceSet = sample(checked, REFERENCE_SAMPLE, rand)
  for (const {N, v} of referenceSet) referenceRight += Number(run(N, v, initialSource, referenceStep) === v % N)
  // Random one-hot scratch states: the machine settles from any start.
  let randomRight = 0
  const randomSet = sample(checked, RANDOM_START_SAMPLE, rand)
  for (const {N, v} of randomSet) {
    const start = new Float32Array(initialSource)
    for (let p = 0; p < lay.L; p++) {
      if (!built.isScratch(p)) continue
      const d = Math.floor(rand() * B)
      const c = Math.floor(rand() * 2)
      built.grid[p]!.digit.forEach((wire, i) => { start[wire] = Number(i === d) })
      built.grid[p]!.borrow.forEach((wire, i) => { start[wire] = Number(i === c) })
    }
    randomRight += Number(run(N, v, start, tensorStep) === v % N)
  }

  // Live operations, as the compiled program packs them, with what the builder wrote and the tags.
  const live = plain(program.wireOrder).slice(circuit.sourceSize)
  const operations = live.map(wire => {
    const op = circuit.operations[wire - circuit.sourceSize]!
    return {
      wire, kind: op.kind, left: op.leftTerms.map(([w2, c]) => [w2, c]), leftBias: op.leftBias,
      gate: op.gateTerms.map(([w2, c]) => [w2, c]), gateBias: op.gateBias, macro: op.macro?.kind ?? null,
      tags: built.tags.get(wire as never) ?? [],
    }
  })
  const counts = {
    cells: lay.L, ticks: lay.depth, sourceCells: circuit.sourceSize, liveOperations: program.valueCount - program.sourceSize,
    builtOperations: circuit.operations.length, layers: program.layers.length,
    storedWeights: program.layers.reduce((sum, layer) => sum + layer.weights.length, 0),
    reglu: program.layers.reduce((sum, layer) => sum + layer.gateRows.length, 0),
    kinds: {SHIFT: lay.kind.filter(x => x === SHIFT).length, SUB: lay.kind.filter(x => x === SUB).length,
      SEL: lay.kind.filter(x => x === SEL).length, inputs: lay.kind.filter(x => x === NIN || x === VIN).length},
  }
  const checks = {
    questions: all.length, checked: checked.length, tensorRight,
    reference: referenceSet.length, referenceRight, randomStart: randomSet.length, randomRight,
    exact: tensorRight === checked.length && referenceRight === referenceSet.length && randomRight === randomSet.length,
  }
  const record = {
    size: {B, w, k}, layout: lay, counts, checks,
    grid: built.grid, nextSource: plain(nextSource), initialSource: plain(initialSource), stateWires, roots,
    program: {
      sourceSize: program.sourceSize, valueCount: program.valueCount, wireOrder: plain(program.wireOrder),
      roots: plain(program.roots),
      layers: program.layers.map(layer => ({
        inputIndices: plain(layer.inputIndices), outputSize: layer.outputSize, bias: plain(layer.bias),
        gateRows: plain(layer.gateRows), rowOffsets: plain(layer.rowOffsets), columns: plain(layer.columns),
        weights: plain(layer.weights),
      })),
    },
    operations,
  }
  mkdirSync(outDir, {recursive: true})
  writeFileSync(join(outDir, `B${B}w${w}k${k}.json`), JSON.stringify(record))
  return {size: [B, w, k], counts, checks}
}

const wanted = process.argv.slice(2).map(arg => arg.split(',').map(Number))
const results = []
for (const [B, w, k] of wanted.length ? wanted : SIZES) {
  const t0 = performance.now()
  const result = compileSize(B!, w!, k!)
  results.push({...result, seconds: (performance.now() - t0) / 1000})
  const {counts: c, checks: x} = result
  console.log(`(${B},${w},${k}) cells ${c.cells} ticks ${c.ticks} live ops ${c.liveOperations} layers ${c.layers} `
    + `weights ${c.storedWeights}: tensor ${x.tensorRight}/${x.checked} reference ${x.referenceRight}/${x.reference} `
    + `random start ${x.randomRight}/${x.randomStart} ${x.exact ? 'EXACT' : 'WRONG'}`)
}
if (!wanted.length) writeFileSync(join(here, '..', 'records', 'compile_js.json'), JSON.stringify(results, null, 1))
