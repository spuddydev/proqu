export const meta = {
  name: 'explore',
  description: 'Map the code a change will touch, one read only Sonnet agent per area, then name what nobody looked at',
  whenToUse: 'The first step of every branch, before the plan is written. Pass the change and the areas to map.',
  phases: [{ title: 'Explore' }, { title: 'Gaps' }],
}

// args: { change: string, areas: [{ key: string, ask: string }], issues?: number[] }
const change = args && args.change
const areas = (args && args.areas) || []
if (!change || areas.length === 0) throw new Error('pass args { change, areas: [{ key, ask }] }')
const issues = (args.issues || []).map(n => `#${n}`).join(', ')

const rules = `Repository: the proqu checkout in the working directory, on its current branch. You are read only: never edit, create or delete files in the repository, never commit, push or write to the network. Run a build or test only when a claim needs it, and only as: timeout 900 uv run ...
The change being planned: ${change}
${issues ? `Issues it closes or touches: ${issues}. Read them and their comments with gh first.` : ''}
Report what is there, not what should be built. Cite every claim with file:line and quote short snippets where the exact text matters.`

const MAP = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'How this area works today, in a few sentences' },
    seams: {
      type: 'array',
      description: 'Every place the change will have to touch or respect',
      items: {
        type: 'object',
        properties: { where: { type: 'string' }, what: { type: 'string' } },
        required: ['where', 'what'],
      },
    },
    tests: { type: 'array', items: { type: 'string' }, description: 'Existing cases and harnesses that cover this area, by file and name' },
    surprises: { type: 'array', items: { type: 'string' }, description: 'Anything that contradicts the issue text or an obvious assumption' },
  },
  required: ['summary', 'seams', 'tests', 'surprises'],
}

phase('Explore')
const maps = await parallel(areas.map(a => () =>
  agent(`${rules}\n\nArea: ${a.key}\n${a.ask}`, { label: `explore:${a.key}`, phase: 'Explore', schema: MAP, model: 'sonnet', effort: 'high' })))

const dead = areas.filter((_, i) => !maps[i]).map(a => a.key)
if (dead.length) log(`no map came back for: ${dead.join(', ')}`)

phase('Gaps')
const gaps = await agent(
  `${rules}\n\nThese areas were mapped: ${areas.map(a => a.key).join(', ')}.\nTheir seams:\n${JSON.stringify(maps.filter(Boolean).map(m => m.seams))}\n\nWhat did nobody look at that the change will reach? Name files, callers, tests and issues outside every map. Do not repeat what is above.`,
  { label: 'gaps', phase: 'Gaps', model: 'sonnet', effort: 'high' })

return { maps: areas.map((a, i) => ({ area: a.key, map: maps[i] })), dead, gaps }
