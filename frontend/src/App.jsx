import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const initialForm = {
  body_type: 'rectangle',
  occasion: 'casual',
  target_market: 'women',
  category: 'top',
  fit: 'regular',
  budget: 2000,
  sustainability: 0,
  color: '',
  material: '',
  style: '',
}
function cleanMarkdown(text) {
  if (!text) return ''

  return text
    // Unescape Markdown headings, bold, italics, and horizontal rules
    .replace(/\\([#*_~`>|])/g, '$1')
    // Remove escaped horizontal rules
    .replace(/\\-{3,}/g, '---')
    .trim()
}
function App() {
  const [form, setForm] = useState(initialForm)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const updateField = (e) => {
    const { name, value } = e.target

    setForm((prev) => ({
      ...prev,
      [name]:
        name === 'budget' || name === 'sustainability'
          ? Number(value)
          : value,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setResults(null)

    try {
      const payload = {
        ...form,
        color: form.color.trim() || null,
        material: form.material.trim() || null,
        style: form.style.trim() || null,
      }

      const response = await fetch('http://127.0.0.1:8000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail
            ? JSON.stringify(data.detail)
            : 'Could not fetch recommendations.'
        )
      }

      setResults(data)
    } catch (err) {
      setError(
        err.message.includes('Failed to fetch')
          ? 'Cannot connect to FitMind AI. Make sure your FastAPI backend is running.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setForm(initialForm)
    setResults(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-[#faf9f6] text-[#25251f]">
      {/* Navigation */}
      <nav className="sticky top-0 z-20 border-b border-stone-200 bg-[#faf9f6]/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <a href="#" className="text-2xl font-black tracking-tight">
            fitmind<span className="text-[#8c9172]">.</span>
          </a>

          <div className="hidden items-center gap-8 text-sm font-medium text-stone-600 md:flex">
            <a href="#discover" className="hover:text-black">
              Discover
            </a>
            <a href="#preferences" className="hover:text-black">
              Your style
            </a>
            <a href="#about" className="hover:text-black">
              About
            </a>
          </div>

          <a
            href="#preferences"
            className="rounded-full bg-[#303329] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[#4b4f40]"
          >
            Find my fit ↗
          </a>
        </div>
      </nav>

      {/* Hero */}
      <header className="mx-auto grid max-w-7xl gap-12 px-6 py-16 md:grid-cols-2 md:items-center md:py-24">
        <div>
          <p className="mb-6 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.25em] text-[#777e5b]">
            <span className="h-2 w-2 rounded-full bg-[#8c9172]" />
            Your personal AI stylist
          </p>

          <h1 className="max-w-xl text-5xl font-medium leading-[1.08] tracking-tight md:text-7xl">
            Fashion that
            <br />
            <span className="font-serif italic text-[#858b6b]">
              feels like you.
            </span>
          </h1>

          <p className="mt-7 max-w-lg text-base leading-8 text-stone-600 md:text-lg">
            Discover pieces that fit your body, your budget, and your
            personality. Let AI do the searching while you focus on being you.
          </p>

          <div className="mt-9 flex flex-wrap gap-4">
            <a
              href="#preferences"
              className="rounded-full bg-[#303329] px-7 py-4 text-sm font-semibold text-white transition hover:bg-[#4b4f40]"
            >
              Build my style profile →
            </a>

            <a
              href="#about"
              className="rounded-full border border-stone-300 px-7 py-4 text-sm font-semibold transition hover:bg-white"
            >
              How it works
            </a>
          </div>

          <div className="mt-12 flex flex-wrap gap-8 border-t border-stone-200 pt-7">
            <div>
              <p className="text-2xl font-semibold">AI</p>
              <p className="mt-1 text-xs text-stone-500">Powered styling</p>
            </div>
            <div>
              <p className="text-2xl font-semibold">Smart</p>
              <p className="mt-1 text-xs text-stone-500">Product matching</p>
            </div>
            <div>
              <p className="text-2xl font-semibold">You</p>
              <p className="mt-1 text-xs text-stone-500">Always the focus</p>
            </div>
          </div>
        </div>

        <div className="relative">
          <div className="absolute -inset-4 rounded-[2.5rem] bg-[#e8e8dd]" />

          <div className="relative flex min-h-[400px] flex-col justify-between overflow-hidden rounded-[2rem] bg-[#d7d9c8] p-8 md:min-h-[520px] md:p-12">
            <div className="relative z-10 flex items-start justify-between">
              <span className="rounded-full bg-white/70 px-4 py-2 text-xs font-semibold text-[#464a3a]">
                STYLE, REIMAGINED
              </span>
              <span className="text-3xl text-[#6d7457]">✳</span>
            </div>

            <div className="relative z-10">
              <p className="font-serif text-5xl italic leading-tight text-[#424735] md:text-6xl">
                Less scrolling.
                <br />
                More wearing.
              </p>
              <p className="mt-5 max-w-xs text-sm leading-6 text-[#62664f]">
                A wardrobe that makes sense for your life, your shape, and
                your own sense of style.
              </p>
            </div>

            <div className="relative z-10 mt-10 flex items-center justify-between">
              <span className="text-xs font-medium tracking-widest text-[#62664f]">
                FITMIND AI · PERSONAL STYLE
              </span>
              <span className="text-3xl text-[#6d7457]">↗</span>
            </div>

            <div className="absolute -right-12 top-24 h-48 w-48 rounded-full border border-white/40" />
            <div className="absolute -right-6 top-32 h-36 w-36 rounded-full border border-white/40" />
          </div>
        </div>
      </header>

      {/* Preferences */}
      <section
        id="preferences"
        className="scroll-mt-24 border-y border-stone-200 bg-white"
      >
        <div className="mx-auto max-w-7xl px-6 py-16 md:py-24">
          <div className="mb-12 max-w-2xl">
            <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
              Your style profile
            </p>
            <h2 className="mt-4 text-4xl font-medium tracking-tight md:text-5xl">
              Tell us what
              <br />
              <span className="font-serif italic text-[#858b6b]">
                you're looking for.
              </span>
            </h2>
            <p className="mt-5 leading-7 text-stone-500">
              A few details help your AI stylist discover clothing from your
              product catalogue that matches your preferences.
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {/* Body type */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Body type
                </label>
                <select
                  name="body_type"
                  value={form.body_type}
                  onChange={updateField}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                >
                  <option value="inverted_triangle">Inverted triangle</option>
                  <option value="pear">Pear</option>
                  <option value="rectangle">Rectangle</option>
                </select>
              </div>

              {/* Occasion */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Occasion
                </label>
                <select
                  name="occasion"
                  value={form.occasion}
                  onChange={updateField}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                >
                  <option value="casual">Casual</option>
                  <option value="formal">Formal</option>
                  <option value="party">Party</option>
                </select>
              </div>

              {/* Target market */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Shop for
                </label>
                <select
                  name="target_market"
                  value={form.target_market}
                  onChange={updateField}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                >
                  <option value="women">Women</option>
                  <option value="men">Men</option>
                  <option value="unisex">Unisex</option>
                </select>
              </div>

              {/* Category */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Clothing category
                </label>
                <select
                  name="category"
                  value={form.category}
                  onChange={updateField}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                >
                  <option value="top">Tops</option>
                  <option value="bottom">Bottoms</option>
                  <option value="one_piece">One-piece</option>
                  <option value="footwear">Footwear</option>
                  <option value="accessory">Accessories</option>
                </select>
              </div>

              {/* Fit */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Preferred fit
                </label>
                <select
                  name="fit"
                  value={form.fit}
                  onChange={updateField}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                >
                  <option value="regular">Regular</option>
                  <option value="oversized">Oversized</option>
                  <option value="relaxed">Relaxed</option>
                  <option value="slim">Slim</option>
                  <option value="skinny">Skinny</option>
                  <option value="straight">Straight</option>
                  <option value="tailored">Tailored</option>
                  <option value="unknown">No preference</option>
                </select>
              </div>

              {/* Budget */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Maximum budget (₹)
                </label>
                <input
                  type="number"
                  name="budget"
                  min="1"
                  value={form.budget}
                  onChange={updateField}
                  required
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                />
              </div>

              {/* Color */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Preferred colour
                </label>
                <input
                  type="text"
                  name="color"
                  value={form.color}
                  onChange={updateField}
                  placeholder="e.g. sage green, black"
                  maxLength={50}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                />
              </div>

              {/* Material */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Preferred material
                </label>
                <input
                  type="text"
                  name="material"
                  value={form.material}
                  onChange={updateField}
                  placeholder="e.g. cotton, linen"
                  maxLength={50}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                />
              </div>

              {/* Style */}
              <div>
                <label className="mb-2 block text-sm font-semibold">
                  Personal style
                </label>
                <input
                  type="text"
                  name="style"
                  value={form.style}
                  onChange={updateField}
                  placeholder="e.g. minimalist, streetwear"
                  maxLength={50}
                  className="w-full rounded-xl border border-stone-200 bg-[#faf9f6] px-4 py-4 outline-none focus:border-[#858b6b]"
                />
              </div>
            </div>

            {/* Sustainability */}
            <div className="mt-10 rounded-2xl bg-[#f5f5ef] p-6 md:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="font-semibold">Sustainability preference</h3>
                  <p className="mt-2 text-sm text-stone-500">
                    Minimum sustainability score from 0 to 10
                  </p>
                </div>
                <span className="rounded-full bg-white px-5 py-3 text-sm font-bold">
                  {form.sustainability} / 10
                </span>
              </div>

              <input
                type="range"
                name="sustainability"
                min="0"
                max="10"
                value={form.sustainability}
                onChange={updateField}
                className="mt-6 w-full accent-[#858b6b]"
              />

              <div className="mt-2 flex justify-between text-xs text-stone-400">
                <span>No minimum</span>
                <span>Higher minimum</span>
              </div>
            </div>

            {/* Buttons */}
            <div className="mt-8 flex flex-wrap gap-4">
              <button
                type="submit"
                disabled={loading}
                className="rounded-full bg-[#303329] px-8 py-4 text-sm font-semibold text-white transition hover:bg-[#4b4f40] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading
                  ? 'Finding your style...'
                  : '✳  Find my recommendations'}
              </button>

              <button
                type="button"
                onClick={resetForm}
                className="rounded-full border border-stone-300 px-8 py-4 text-sm font-semibold transition hover:bg-stone-100"
              >
                Reset preferences
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-8 rounded-2xl border border-red-200 bg-red-50 p-5 text-sm leading-6 text-red-700">
              <strong>Something went wrong.</strong>
              <p className="mt-1">{error}</p>
            </div>
          )}
        </div>
      </section>

      {/* Results */}
      {results && (
        <section id="discover" className="scroll-mt-24">
          <div className="mx-auto max-w-7xl px-6 py-16 md:py-24">
            <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
              Curated for you
            </p>

            <h2 className="mt-4 text-4xl font-medium tracking-tight md:text-5xl">
              Your style, <span className="font-serif italic">your picks.</span>
            </h2>

            {results.recommendation && (
              <div className="mt-8 rounded-2xl border border-[#e1e2d6] bg-[#f2f3eb] p-6 md:p-8">
                <p className="text-xs font-bold uppercase tracking-widest text-[#858b6b]">
                  Your AI stylist says
                </p>

                <div className="mt-6 break-words text-base leading-8 text-stone-700">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({ children }) => (
                        <h1 className="mb-4 mt-6 text-3xl font-medium tracking-tight text-stone-900 first:mt-0">
                          {children}
                        </h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="mb-3 mt-8 text-2xl font-medium tracking-tight text-stone-900 first:mt-0">
                          {children}
                        </h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="mb-2 mt-6 text-lg font-semibold text-stone-900">
                          {children}
                        </h3>
                      ),
                      p: ({ children }) => (
                        <p className="mb-4 leading-8 last:mb-0">{children}</p>
                      ),
                      strong: ({ children }) => (
                        <strong className="font-semibold text-stone-900">
                          {children}
                        </strong>
                      ),
                      em: ({ children }) => (
                        <em className="font-serif italic text-[#737a57]">
                          {children}
                        </em>
                      ),
                      ul: ({ children }) => (
                        <ul className="mb-5 ml-6 list-disc space-y-2 marker:text-[#858b6b]">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="mb-5 ml-6 list-decimal space-y-2 marker:font-semibold marker:text-[#858b6b]">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => (
                        <li className="pl-1 leading-7">{children}</li>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="my-6 border-l-4 border-[#858b6b] bg-white/60 py-3 pl-5 pr-4 italic text-stone-600">
                          {children}
                        </blockquote>
                      ),
                      hr: () => <hr className="my-8 border-[#d9dbce]" />,
                      table: ({ children }) => (
                        <div className="my-6 overflow-x-auto rounded-xl border border-[#e1e2d6] bg-white">
                          <table className="w-full border-collapse text-left text-sm">
                            {children}
                          </table>
                        </div>
                      ),
                      thead: ({ children }) => (
                        <thead className="bg-[#e9ebdf] text-stone-900">
                          {children}
                        </thead>
                      ),
                      th: ({ children }) => (
                        <th className="border-b border-[#e1e2d6] px-4 py-3 font-semibold">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border-b border-[#e1e2d6] px-4 py-3">
                          {children}
                        </td>
                      ),
                      a: ({ children, href }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noreferrer"
                          className="font-medium text-[#62694c] underline underline-offset-4 hover:text-[#303329]"
                        >
                          {children}
                        </a>
                      ),
                      code: ({ children, className }) =>
                        className ? (
                          <code className="block overflow-x-auto rounded-xl bg-[#303329] p-4 text-sm text-stone-100">
                            {children}
                          </code>
                        ) : (
                          <code className="rounded-md bg-white px-2 py-1 text-sm text-[#62694c]">
                            {children}
                          </code>
                        ),
                    }}
                  >
                    {results.recommendation}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {Array.isArray(results.products) &&
              results.products.length > 0 && (
                <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                  {results.products.map((product, index) => (
                    <article
                      key={product.product_id ?? product.id ?? index}
                      className="overflow-hidden rounded-2xl border border-stone-200 bg-white transition hover:-translate-y-1 hover:shadow-lg"
                    >
                      <div className="flex aspect-[4/3] items-center justify-center bg-[#e9e9df]">
                        <span className="text-6xl text-[#a2a58c]">✳</span>
                      </div>

                      <div className="p-6">
                        <div className="flex items-start justify-between gap-3">
                          <h3 className="font-semibold leading-6">
                            {product.product_name ??
                              product.name ??
                              'Fashion product'}
                          </h3>

                          {product.price != null && (
                            <span className="shrink-0 font-semibold">
                              ₹{product.price}
                            </span>
                          )}
                        </div>

                        {product.category && (
                          <p className="mt-2 text-sm capitalize text-stone-500">
                            {product.category}
                          </p>
                        )}

                        {product.color && (
                          <p className="mt-2 text-sm text-stone-500">
                            Colour: {product.color}
                          </p>
                        )}

                        {product.fit && (
                          <p className="mt-1 text-sm text-stone-500">
                            Fit: {product.fit}
                          </p>
                        )}

                        {product.semantic_score != null && (
                          <div className="mt-4 inline-block rounded-full bg-[#f0f1e9] px-3 py-2 text-xs font-medium text-[#62694c]">
                            Retrieval similarity: {Number(product.semantic_score).toFixed(3)}
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              )}

            {Array.isArray(results.products) &&
              results.products.length === 0 && (
                <div className="mt-10 rounded-2xl bg-white p-8 text-stone-600">
                  No matching products were returned. Try adjusting your
                  preferences or budget.
                </div>
              )}
          </div>
        </section>
      )}

      {/* About */}
      <section
        id="about"
        className="border-t border-stone-200 bg-[#f0f0e8]"
      >
        <div className="mx-auto grid max-w-7xl gap-12 px-6 py-16 md:grid-cols-2 md:items-center md:py-20">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
              The FitMind approach
            </p>
            <h2 className="mt-4 text-4xl font-medium leading-tight md:text-5xl">
              Your wardrobe,
              <br />
              <span className="font-serif italic text-[#858b6b]">
                with a little intelligence.
              </span>
            </h2>
          </div>

          <div className="space-y-6 text-sm leading-7 text-stone-600">
            <p>
              FitMind AI combines your preferences with semantic product
              search to discover fashion from a curated catalogue.
            </p>
            <p>
              Your selections help filter products by category, occasion,
              budget, fit, and sustainability preferences.
            </p>
            <p>
              An AI stylist then provides additional styling suggestions
              grounded in the retrieved products.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[#303329] text-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-8 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xl font-black tracking-tight">
            fitmind<span className="text-[#b9c09c]">.</span>
          </p>
          <p className="text-xs text-stone-400">
            AI-powered fashion discovery · Built with care
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App