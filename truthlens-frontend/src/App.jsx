import { useState } from 'react'
import {
  ShieldCheck, Search, FileText, Activity, ArrowRight,
  LoaderCircle, AlertCircle, CheckCircle2, XCircle,
  CircleHelp, Sparkles, RotateCcw
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000'

const verdictStyles = {
  SUPPORTED: {
    label: 'Supported',
    icon: CheckCircle2,
    color: 'text-emerald-400',
    border: 'border-emerald-400/25',
    background: 'bg-emerald-400/10',
  },
  CONTRADICTED: {
    label: 'Contradicted',
    icon: XCircle,
    color: 'text-rose-400',
    border: 'border-rose-400/25',
    background: 'bg-rose-400/10',
  },
  INSUFFICIENT_EVIDENCE: {
    label: 'Insufficient evidence',
    icon: CircleHelp,
    color: 'text-amber-300',
    border: 'border-amber-300/25',
    background: 'bg-amber-300/10',
  },
}

function formatVerdict(value = '') {
  return value.toUpperCase().replaceAll(' ', '_').replaceAll('-', '_')
}

function App() {
  const [claim, setClaim] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function verifyClaim(event) {
    event.preventDefault()

    const text = claim.trim()
    if (!text) {
      setError('Please enter a claim or social-media post first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Verification failed.')
      }

      if (data.success === false) {
        throw new Error(data.error || 'The verification could not be completed.')
      }

      setResult(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to the TruthLens API. Make sure your Python backend is running on port 8000.'
          : err.message || 'Something went wrong.'
      )
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setClaim('')
    setResult(null)
    setError('')
  }

  const verdict = formatVerdict(result?.verdict)
  const style = verdictStyles[verdict]
  const VerdictIcon = style?.icon || CircleHelp
  const confidence = Number(result?.confidence)
  const validConfidence = Number.isFinite(confidence) && confidence >= 0 && confidence <= 1

  return (
    <div className="min-h-screen bg-[#090d18] text-slate-100">
      <header className="sticky top-0 z-10 border-b border-white/[0.08] bg-[#090d18]/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
          <a href="#" className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/15 text-indigo-300 ring-1 ring-indigo-400/30">
              <ShieldCheck size={23} />
            </div>
            <div>
              <div className="text-lg font-bold tracking-tight">TruthLens<span className="text-indigo-400">.</span></div>
              <div className="text-[11px] tracking-widest text-slate-500">EVIDENCE-BASED VERIFICATION</div>
            </div>
          </a>

          <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/[0.06] px-3 py-1.5 text-xs text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            API integration
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-12 sm:px-8 sm:py-16">
        <section className="mx-auto max-w-3xl text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-400/20 bg-indigo-400/[0.07] px-3 py-1.5 text-xs font-medium text-indigo-300">
            <Sparkles size={14} /> AI-powered claim analysis
          </div>

          <h1 className="text-4xl font-bold leading-tight tracking-tight sm:text-6xl">
            Look beyond the
            <span className="block bg-gradient-to-r from-indigo-300 via-violet-300 to-cyan-300 bg-clip-text text-transparent">
              headline.
            </span>
          </h1>

          <p className="mx-auto mt-5 max-w-xl text-sm leading-7 text-slate-400 sm:text-base">
            Verify claims with evidence from historical fact-checks and real-time web sources.
            Get a verdict and understand the reasoning behind it.
          </p>
        </section>

        <section className="mx-auto mt-10 max-w-3xl">
          <form onSubmit={verifyClaim} className="rounded-2xl border border-white/10 bg-[#111827] p-4 shadow-2xl shadow-black/20 sm:p-6">
            <label htmlFor="claim" className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-200">
              <FileText size={17} className="text-indigo-300" />
              Paste a claim or social-media post
            </label>

            <textarea
              id="claim"
              value={claim}
              onChange={(event) => setClaim(event.target.value)}
              placeholder="Example: Antibiotics kill viruses..."
              maxLength={10000}
              rows={5}
              className="w-full resize-y rounded-xl border border-white/10 bg-[#090d18] px-4 py-4 text-sm leading-6 text-slate-100 placeholder:text-slate-600 focus:border-indigo-400/60 focus:outline-none"
            />

            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs text-slate-500">{claim.length}/10,000 characters</span>
              <button
                type="submit"
                disabled={loading || !claim.trim()}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading
                  ? <><LoaderCircle size={17} className="animate-spin" /> Analyzing...</>
                  : <>Verify claim <ArrowRight size={17} /></>}
              </button>
            </div>
          </form>

          {error && (
            <div role="alert" className="mt-4 flex items-start gap-3 rounded-xl border border-rose-400/20 bg-rose-400/[0.07] p-4 text-sm text-rose-300">
              <AlertCircle size={19} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </section>

        {loading && (
          <div className="mx-auto mt-10 max-w-3xl rounded-2xl border border-white/10 bg-[#111827] p-8 text-center">
            <LoaderCircle size={30} className="mx-auto animate-spin text-indigo-300" />
            <p className="mt-4 font-medium">Analyzing your claim</p>
            <p className="mt-2 text-sm text-slate-500">Retrieving evidence and evaluating the claim. This may take a little while.</p>
          </div>
        )}

        {result && !loading && (
          <section aria-live="polite" className="mx-auto mt-10 max-w-4xl space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-300">Analysis report</p>
                <h2 className="mt-1 text-2xl font-bold">Verification results</h2>
              </div>
              <button onClick={reset} className="inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-300 transition hover:bg-white/5">
                <RotateCcw size={15} /> New claim
              </button>
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-7">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Claim analyzed</p>
              <p className="mt-3 text-lg leading-8 text-slate-100">{result.claim || claim}</p>

              <div className={`mt-6 flex flex-col justify-between gap-5 rounded-xl border p-5 sm:flex-row sm:items-center ${style?.border || 'border-white/10'} ${style?.background || 'bg-white/[0.03]'}`}>
                <div className="flex items-center gap-3">
                  <VerdictIcon size={29} className={style?.color || 'text-slate-300'} />
                  <div>
                    <p className="text-xs text-slate-400">Final verdict</p>
                    <p className={`mt-1 text-xl font-bold ${style?.color || 'text-slate-200'}`}>
                      {style?.label || result.verdict || 'Unknown'}
                    </p>
                  </div>
                </div>

                <div className="min-w-40">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span className="text-slate-400">Model confidence</span>
                    <span className="font-semibold text-slate-200">
                      {validConfidence ? `${(confidence * 100).toFixed(1)}%` : 'Not established'}
                    </span>
                  </div>
                  {validConfidence && (
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/10">
                      <div className="h-full rounded-full bg-indigo-400 transition-all" style={{ width: `${confidence * 100}%` }} />
                    </div>
                  )}
                  <p className="mt-2 max-w-64 text-xs text-slate-500">Model score, not a guarantee of factual accuracy.</p>
                </div>
              </div>
            </div>

            {Array.isArray(result.atomic_claims) && result.atomic_claims.length > 0 && (
              <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
                <div className="flex items-center gap-2">
                  <Activity size={18} className="text-indigo-300" />
                  <h3 className="font-semibold">Atomic claims</h3>
                </div>
                <ul className="mt-4 space-y-3">
                  {result.atomic_claims.map((item, index) => (
                    <li key={`${index}-${item}`} className="flex gap-3 rounded-xl bg-white/[0.03] p-3 text-sm leading-6 text-slate-300">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-indigo-400/10 text-xs text-indigo-300">{index + 1}</span>
                      <span>{typeof item === 'string' ? item : JSON.stringify(item)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {Array.isArray(result.evidence) && result.evidence.length > 0 && (
              <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ShieldCheck size={18} className="text-indigo-300" />
                    <h3 className="font-semibold">Retrieved Evidence & XLM-R NLI Evaluation</h3>
                  </div>
                  <span className="text-xs text-slate-400">{result.evidence.length} evidence sources ranked</span>
                </div>
                <div className="mt-4 space-y-3">
                  {result.evidence.map((item, index) => (
                    <div key={index} className="rounded-xl border border-white/5 bg-white/[0.02] p-4 text-xs leading-5">
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-2">
                        <div className="flex items-center gap-2">
                          <span className={`rounded px-2 py-0.5 font-semibold text-[10px] ${
                            item.source === 'HISTORICAL' ? 'bg-blue-500/20 text-blue-300' : 'bg-emerald-500/20 text-emerald-300'
                          }`}>
                            {item.source === 'HISTORICAL' ? 'FactDrill Corpus' : 'Real-Time Web'}
                          </span>
                          <span className="font-medium text-slate-300">{item.website || 'Source'}</span>
                          {item.publish_date && <span className="text-slate-500">· {item.publish_date.slice(0, 10)}</span>}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`rounded px-2 py-0.5 font-bold text-[10px] ${
                            item.nli_label === 'ENTAILMENT' ? 'bg-emerald-500/20 text-emerald-400' :
                            item.nli_label === 'CONTRADICTION' ? 'bg-rose-500/20 text-rose-400' :
                            'bg-amber-500/20 text-amber-300'
                          }`}>
                            XLM-R: {item.nli_label} ({(item.nli_confidence * 100).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                      {item.title && <p className="mt-2 font-semibold text-slate-200">{item.title}</p>}
                      <p className="mt-1 text-slate-400 line-clamp-3">{item.snippet}</p>
                      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-500">
                        <span>Rank: #{index + 1}</span>
                        <span>Retrieval: {item.retrieval_score}</span>
                        <span>Temporal: {item.temporal_score}</span>
                        <span>Credibility: {item.credibility}x</span>
                        <span>Weight: {item.evidence_weight}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-2xl border border-white/10 bg-[#111827] p-5 sm:p-6">
              <div className="flex items-center gap-2">
                <Search size={18} className="text-indigo-300" />
                <h3 className="font-semibold">RAG explanation</h3>
              </div>
              <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-slate-300">
                {result.explanation || 'No explanation was returned by the backend.'}
              </p>
            </div>

            <p className="px-1 text-xs leading-5 text-slate-500">
              TruthLens provides AI-generated analysis to assist fact-checking. Verify important claims against original, reliable sources.
            </p>
          </section>
        )}

        {!result && !loading && (
          <section className="mx-auto mt-12 grid max-w-4xl gap-4 sm:grid-cols-3">
            {[
              { icon: Search, title: 'Evidence retrieval', description: 'Search across historical fact-checks and real-time web results.' },
              { icon: ShieldCheck, title: 'Claim assessment', description: 'Evaluate evidence using multilingual natural language inference.' },
              { icon: FileText, title: 'Clear explanations', description: 'Understand the reasoning behind the generated verdict.' },
            ].map(({ icon: Icon, title, description }) => (
              <div key={title} className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-400/10 text-indigo-300">
                  <Icon size={20} />
                </div>
                <h3 className="mt-4 font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">{description}</p>
              </div>
            ))}
          </section>
        )}

        <footer className="mt-16 border-t border-white/[0.08] pt-6 text-center text-xs text-slate-600">
          TruthLens · Evidence-aware misinformation verification
        </footer>
      </main>
    </div>
  )
}

export default App