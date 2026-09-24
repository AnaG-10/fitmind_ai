import ProductCard from "../components/ProductCard";
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion } from 'framer-motion'
import {
  Sparkles,
  Shirt,
  LoaderCircle,
  ArrowRight,
  RotateCcw,
  Leaf,
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/recommend'

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

const options = {
  body_type: [
    { value: 'inverted_triangle', label: 'Inverted Triangle' },
    { value: 'pear', label: 'Pear' },
    { value: 'rectangle', label: 'Rectangle' },
  ],
  occasion: [
    { value: 'casual', label: 'Casual' },
    { value: 'formal', label: 'Formal' },
    { value: 'party', label: 'Party' },
  ],
  target_market: [
    { value: 'women', label: 'Women' },
    { value: 'men', label: 'Men' },
    { value: 'unisex', label: 'Unisex' },
  ],
  category: [
    { value: 'top', label: 'Tops' },
    { value: 'bottom', label: 'Bottoms' },
    { value: 'one_piece', label: 'One Piece' },
    { value: 'footwear', label: 'Footwear' },
    { value: 'accessory', label: 'Accessories' },
  ],
  fit: [
    { value: 'regular', label: 'Regular' },
    { value: 'oversized', label: 'Oversized' },
    { value: 'relaxed', label: 'Relaxed' },
    { value: 'slim', label: 'Slim' },
    { value: 'skinny', label: 'Skinny' },
    { value: 'straight', label: 'Straight' },
    { value: 'tailored', label: 'Tailored' },
    { value: 'unknown', label: 'No preference' },
  ],
}

function Field({ label, children }) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-semibold uppercase tracking-[0.15em] text-[#77776e]">
        {label}
      </label>
      {children}
    </div>
  )
}

function SelectField({ label, name, value, onChange }) {
  return (
    <Field label={label}>
      <select
        name={name}
        value={value}
        onChange={onChange}
        className="w-full rounded-xl border border-[#e4e2da] bg-white px-4 py-3.5 text-sm outline-none transition focus:border-[#777b58] focus:ring-2 focus:ring-[#777b58]/10"
      >
        {options[name].map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  )
}

export default function FindMyFit() {
  const [form, setForm] = useState(initialForm)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  function handleChange(event) {
    const { name, value } = event.target

    setForm((prev) => ({
      ...prev,
      [name]:
        name === 'budget' || name === 'sustainability'
          ? Number(value)
          : value,
    }))
  }

  async function handleSubmit(event) {
    event.preventDefault()

    setLoading(true)
    setError('')
    setResults(null)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(form),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail
            ? typeof data.detail === 'string'
              ? data.detail
              : JSON.stringify(data.detail)
            : 'Something went wrong. Please try again.'
        )
      }

      setResults(data)
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to FitMind AI. Please make sure your FastAPI backend is running on port 8000.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  function resetForm() {
    setForm(initialForm)
    setResults(null)
    setError('')
  }

  return (
    <main className="min-h-screen bg-[#faf9f6] px-5 py-12 sm:px-8 lg:px-16">
      <div className="mx-auto max-w-7xl">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-12"
        >
          <div className="mb-5 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-[#777b58]">
            <Sparkles size={15} />
            AI-powered personal styling
          </div>

          <h1 className="max-w-3xl text-4xl font-light leading-tight tracking-tight text-[#25251f] sm:text-6xl">
            Find your fit.
            <br />
            <span className="font-serif italic text-[#777b58]">
              Define your style.
            </span>
          </h1>

          <p className="mt-5 max-w-xl text-sm leading-7 text-[#77776e] sm:text-base">
            Tell us what you love, what you need, and what feels like you.
            FitMind AI will curate fashion recommendations around your
            preferences.
          </p>
        </motion.div>

        <div className="grid items-start gap-10 lg:grid-cols-[0.9fr_1.1fr]">

          {/* Form */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="rounded-3xl border border-[#e9e7df] bg-white p-6 shadow-sm sm:p-9"
          >
            <div className="mb-8 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-medium">Your style profile</h2>
                <p className="mt-1 text-sm text-[#88887e]">
                  Personalize your recommendations
                </p>
              </div>

              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[#f1f1e9] text-[#777b58]">
                <Shirt size={20} />
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">

              <div className="grid gap-5 sm:grid-cols-2">
                <SelectField
                  label="Body type"
                  name="body_type"
                  value={form.body_type}
                  onChange={handleChange}
                />

                <SelectField
                  label="Shopping for"
                  name="target_market"
                  value={form.target_market}
                  onChange={handleChange}
                />

                <SelectField
                  label="Occasion"
                  name="occasion"
                  value={form.occasion}
                  onChange={handleChange}
                />

                <SelectField
                  label="Category"
                  name="category"
                  value={form.category}
                  onChange={handleChange}
                />

                <div className="sm:col-span-2">
                  <SelectField
                    label="Preferred fit"
                    name="fit"
                    value={form.fit}
                    onChange={handleChange}
                  />
                </div>
              </div>

              {/* Budget */}
              <Field label="Your budget (₹)">
                <input
                  type="number"
                  name="budget"
                  min="1"
                  value={form.budget}
                  onChange={handleChange}
                  required
                  className="w-full rounded-xl border border-[#e4e2da] bg-white px-4 py-3.5 text-sm outline-none transition focus:border-[#777b58] focus:ring-2 focus:ring-[#777b58]/10"
                />
              </Field>

              {/* Sustainability */}
              <Field label="Sustainability preference">
                <div className="rounded-xl border border-[#e4e2da] bg-[#faf9f6] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="flex items-center gap-2 text-sm text-[#66665c]">
                      <Leaf size={16} className="text-[#777b58]" />
                      Sustainability priority
                    </span>
                    <span className="text-sm font-semibold text-[#777b58]">
                      {form.sustainability}/10
                    </span>
                  </div>

                  <input
                    type="range"
                    name="sustainability"
                    min="0"
                    max="10"
                    step="1"
                    value={form.sustainability}
                    onChange={handleChange}
                    className="w-full accent-[#777b58]"
                  />

                  <div className="mt-1 flex justify-between text-xs text-[#99998e]">
                    <span>No preference</span>
                    <span>High priority</span>
                  </div>
                </div>
              </Field>

              {/* Optional preferences */}
              <div className="space-y-5 border-t border-[#eeece5] pt-6">
                <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#77776e]">
                  Additional preferences
                </p>

                <Field label="Preferred color">
                  <input
                    type="text"
                    name="color"
                    value={form.color}
                    onChange={handleChange}
                    placeholder="e.g. olive, beige, black"
                    maxLength={50}
                    className="w-full rounded-xl border border-[#e4e2da] px-4 py-3.5 text-sm outline-none transition focus:border-[#777b58] focus:ring-2 focus:ring-[#777b58]/10"
                  />
                </Field>

                <Field label="Preferred material">
                  <input
                    type="text"
                    name="material"
                    value={form.material}
                    onChange={handleChange}
                    placeholder="e.g. cotton, linen, denim"
                    maxLength={50}
                    className="w-full rounded-xl border border-[#e4e2da] px-4 py-3.5 text-sm outline-none transition focus:border-[#777b58] focus:ring-2 focus:ring-[#777b58]/10"
                  />
                </Field>

                <Field label="Your personal style">
                  <input
                    type="text"
                    name="style"
                    value={form.style}
                    onChange={handleChange}
                    placeholder="e.g. minimal, streetwear, chic"
                    maxLength={50}
                    className="w-full rounded-xl border border-[#e4e2da] px-4 py-3.5 text-sm outline-none transition focus:border-[#777b58] focus:ring-2 focus:ring-[#777b58]/10"
                  />
                </Field>
              </div>

              {/* Error */}
              {error && (
                <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">
                  {error}
                </div>
              )}

              {/* Buttons */}
              <div className="flex flex-col gap-3 pt-2 sm:flex-row">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex flex-1 items-center justify-center gap-3 rounded-full bg-[#777b58] px-6 py-4 text-sm font-medium text-white transition hover:bg-[#626647] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <LoaderCircle size={18} className="animate-spin" />
                      Finding your fit...
                    </>
                  ) : (
                    <>
                      Find my fit
                      <ArrowRight size={18} />
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={resetForm}
                  className="flex items-center justify-center gap-2 rounded-full border border-[#e4e2da] px-5 py-4 text-sm text-[#66665c] transition hover:bg-[#f6f5f0]"
                >
                  <RotateCcw size={16} />
                  Reset
                </button>
              </div>
            </form>
          </motion.section>

          {/* Results */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="min-h-[450px] rounded-3xl border border-[#e9e7df] bg-[#f1f0e9] p-6 sm:p-9"
          >
            <div className="mb-8 flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-[#777b58]">
                <Sparkles size={20} />
              </div>

              <div>
                <h2 className="text-xl font-medium">Your curated edit</h2>
                <p className="mt-1 text-sm text-[#88887e]">
                  Made for your preferences
                </p>
              </div>
            </div>

            {!results && !loading && (
              <div className="flex min-h-[300px] flex-col items-center justify-center px-4 text-center">
                <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-white text-[#777b58]">
                  <Shirt size={32} strokeWidth={1.2} />
                </div>

                <h3 className="text-2xl font-light">
                  Your next favorite
                  <br />
                  <span className="font-serif italic text-[#777b58]">
                    starts here.
                  </span>
                </h3>

                <p className="mt-4 max-w-xs text-sm leading-6 text-[#88887e]">
                  Complete your style profile and let FitMind AI find
                  recommendations tailored to you.
                </p>
              </div>
            )}

            {loading && (
              <div className="flex min-h-[300px] flex-col items-center justify-center text-center">
                <LoaderCircle
                  size={40}
                  className="animate-spin text-[#777b58]"
                />

                <h3 className="mt-6 text-xl font-medium">
                  Curating your style...
                </h3>

                <p className="mt-3 max-w-xs text-sm leading-6 text-[#88887e]">
                  Our AI is matching your preferences with products
                  from the collection.
                </p>
              </div>
            )}

            {results && (
              <div className="space-y-6">
                <div className="rounded-2xl bg-white p-5 sm:p-7">
                  <div className="mb-5 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.15em] text-[#777b58]">
                    <Sparkles size={15} />
                    FitMind AI recommendations
                  </div>

                  <div className="prose prose-sm max-w-none break-words prose-headings:font-medium prose-headings:text-[#25251f] prose-p:leading-7 prose-p:text-[#66665c] prose-strong:text-[#25251f] prose-li:text-[#66665c] prose-a:text-[#777b58]">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {typeof results.recommendation === 'string'
                        ? results.recommendation
                        : JSON.stringify(
                            results.recommendation ?? results,
                            null,
                            2
                          )}
                    </ReactMarkdown>
                  </div>
                </div>


                {/* Product recommendations */}
                {results.products?.length > 0 && (
                  <div className="space-y-5">
                    <div className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#777b58]">
                        Your personalized edit
                      </p>

                      <h3 className="text-2xl font-light tracking-tight text-[#25251f]">
                        Pieces picked for you.
                      </h3>

                      <p className="text-sm leading-6 text-[#88887e]">
                        Explore real products from the FitMind catalog,
                        matched to your preferences.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                      {results.products.map((product, index) => (
                        <ProductCard
                          key={product.item_id}
                          product={product}
                          index={index}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => {
                    setResults(null)
                    setError('')
                  }}
                  className="flex items-center gap-2 text-sm font-medium text-[#777b58] transition hover:text-[#50543a]"
                >
                  Try different preferences
                  <ArrowRight size={16} />
                </button>
              </div>
            )}
          </motion.section>
        </div>

        <div className="mt-10 text-center text-xs tracking-wide text-[#99998e]">
          FITMIND AI · YOUR PERSONAL STYLE COMPANION
        </div>
      </div>
    </main>
  )
}