export const meta = {
  name: 'review',
  description: 'Review a commit range through several lenses, try to refute every finding, and on a deep pass ask what was missed',
  whenToUse: 'A light round after each block of five commits, and the deep pass after the last commit. Never start one while another is running.',
  phases: [{ title: 'Tests' }, { title: 'Review' }, { title: 'Verify' }, { title: 'Completeness' }],
}

// args: { base: string, depth: 'light' | 'deep', change: string, plan?: string, lenses?: [{ key, ask }] }
const base = args && args.base
const deep = args && args.depth === 'deep'
if (!base || !args.change) throw new Error('pass args { base, depth, change }')

const LIGHT = [
  { key: 'logic', ask: 'Logic: wrong conditions, missed states or arms, answers that disagree with what the change set out to do.' },
  { key: 'bugs', ask: 'Bugs: unhandled failure, lifetime and teardown mistakes, races, anything that can hang or leak.' },
  { key: 'tests', ask: 'Tests that would still pass with the change undone, timing that could flake under load, and changed behaviour with no case. Back a claim by mutating a scratch copy.' },
]
const DEEP_EXTRA = [
  { key: 'contracts', ask: 'Comments and interface docs that promise behaviour the code does not perform, and comments that narrate rather than explain why.' },
  { key: 'rules', ask: 'The house rules in CLAUDE.md, and its prose rules in comments and docs.' },
]
const lenses = args.lenses || (deep ? LIGHT.concat(DEEP_EXTRA) : LIGHT)

const rules = `Repository: the proqu checkout in the working directory. You never mutate the work: no edits, commits or pushes in the repository and no network writes. To try a mutation, copy the tree to a scratch directory first. Run builds and tests only as: timeout 900 uv run ...
Scope: git diff ${base}...HEAD and git log ${base}..HEAD.
The change: ${args.change}${args.plan ? `\nThe plan: ${args.plan}` : ''}`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'number' },
          claim: { type: 'string' },
          evidence: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'file', 'claim', 'evidence', 'severity'],
      },
    },
  },
  required: ['findings'],
}
const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reasoning: { type: 'string' } },
  required: ['refuted', 'reasoning'],
}

phase('Tests')
const suite = await agent(
  `${rules}\nRun the suite: timeout 900 uv run pytest, then uvx pre-commit run --all-files. Report pass or fail with the failing output.`,
  { label: 'suite', phase: 'Tests', model: 'sonnet', effort: 'medium' })

const results = await pipeline(
  lenses,
  lens => agent(
    `${rules}\nReview through one lens only.\n${lens.ask}\nReport only what you can evidence from the tree. No style preferences and no speculation.`,
    { label: `review:${lens.key}`, phase: 'Review', schema: FINDINGS, model: 'sonnet', effort: 'high' }),
  review => parallel((review ? review.findings : []).map(f => () =>
    agent(
      `${rules}\nTry to REFUTE this claim.\n\nClaim: ${f.claim}\nAt: ${f.file}:${f.line || ''}\nEvidence offered: ${f.evidence}\n\nCheck it against the actual code. When it is about what a tool or test reports, run it in a scratch copy and judge from the output. Default to refuted when uncertain or when the claim rests on a misreading.`,
      { label: `verify:${f.file}`, phase: 'Verify', schema: VERDICT, model: 'sonnet', effort: 'high' },
    ).then(v => ({ ...f, verdict: v })))),
)

const dead = lenses.filter((_, i) => results[i] === null).map(l => l.key)
if (dead.length) log(`lenses that returned nothing: ${dead.join(', ')}. This is not a clean review.`)
const all = results.flat().filter(Boolean)

let missed = null
if (deep) {
  phase('Completeness')
  missed = await agent(
    `${rules}\nA review reported: ${JSON.stringify(all.map(f => f.title))}\nWhat was NOT examined? Files in the diff no lens covered, claims nobody ran, and items the plan promised that are not done. Do not repeat the findings above.`,
    { label: 'completeness', phase: 'Completeness', schema: FINDINGS, model: 'sonnet', effort: 'high' })
}

return {
  suite,
  dead,
  confirmed: all.filter(f => f.verdict && !f.verdict.refuted),
  refuted: all.filter(f => f.verdict && f.verdict.refuted).map(f => ({ title: f.title, why: f.verdict.reasoning })),
  unverified: all.filter(f => !f.verdict).map(f => f.title),
  missed: missed ? missed.findings : null,
}
