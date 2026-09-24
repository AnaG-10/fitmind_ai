
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight, Check } from 'lucide-react'

const styleOptions = [
  'Minimalist',
  'Casual',
  'Streetwear',
  'Classic',
  'Bohemian',
  'Contemporary',
]

const colorOptions = [
  'Neutrals',
  'Earth tones',
  'Black & white',
  'Pastels',
  'Bold colours',
]

export default function YourStyle() {
  const [selectedStyles, setSelectedStyles] = useState(() => {
    return JSON.parse(localStorage.getItem('fitmind_styles') || '[]')
  })

  const [selectedColors, setSelectedColors] = useState(() => {
    return JSON.parse(localStorage.getItem('fitmind_colors') || '[]')
  })

  const [budget, setBudget] = useState(
    () => localStorage.getItem('fitmind_budget') || '2000'
  )

  const toggleOption = (value, selected, setter, key) => {
    const updated = selected.includes(value)
      ? selected.filter((item) => item !== value)
      : [...selected, value]

    setter(updated)
    localStorage.setItem(key, JSON.stringify(updated))
  }

  const saveProfile = () => {
    localStorage.setItem('fitmind_styles', JSON.stringify(selectedStyles))
    localStorage.setItem('fitmind_colors', JSON.stringify(selectedColors))
    localStorage.setItem('fitmind_budget', budget)
  }

  return (
    <main className="min-h-screen bg-[#faf9f6]">

      <section className="mx-auto max-w-5xl px-6 py-20 md:py-28">

        <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
          Your personal style profile
        </p>

        <h1 className="mt-6 text-5xl font-medium leading-tight md:text-7xl">
          Your style.
          <br />
          <span className="font-serif italic text-[#858b6b]">
            Your rules.
          </span>
        </h1>

        <p className="mt-6 max-w-xl leading-8 text-stone-600">
          Tell us what you love. Your preferences help FitMind
          discover products that match your taste.
        </p>

        {/* STYLE */}

        <div className="mt-16">

          <h2 className="text-2xl font-medium">
            How would you describe your style?
          </h2>

          <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-3">

            {styleOptions.map((style) => {
              const active = selectedStyles.includes(style)

              return (
                <button
                  key={style}
                  onClick={() =>
                    toggleOption(
                      style,
                      selectedStyles,
                      setSelectedStyles,
                      'fitmind_styles'
                    )
                  }
                  className={`flex min-h-24 items-center justify-between rounded-2xl border p-5 text-left transition ${
                    active
                      ? 'border-[#858b6b] bg-[#e9ebdf]'
                      : 'border-stone-200 bg-white hover:border-[#858b6b]'
                  }`}
                >
                  <span className="font-medium">{style}</span>

                  {active && (
                    <Check size={18} className="text-[#62694c]" />
                  )}
                </button>
              )
            })}

          </div>

        </div>

        {/* COLOURS */}

        <div className="mt-14">

          <h2 className="text-2xl font-medium">
            Which colours do you gravitate towards?
          </h2>

          <div className="mt-6 flex flex-wrap gap-3">

            {colorOptions.map((color) => {
              const active = selectedColors.includes(color)

              return (
                <button
                  key={color}
                  onClick={() =>
                    toggleOption(
                      color,
                      selectedColors,
                      setSelectedColors,
                      'fitmind_colors'
                    )
                  }
                  className={`rounded-full border px-6 py-3 text-sm transition ${
                    active
                      ? 'border-[#303329] bg-[#303329] text-white'
                      : 'border-stone-300 bg-white hover:border-[#858b6b]'
                  }`}
                >
                  {color}
                </button>
              )
            })}

          </div>

        </div>

        {/* BUDGET */}

        <div className="mt-14 max-w-md">

          <label className="text-2xl font-medium">
            What's your shopping budget?
          </label>

          <p className="mt-3 text-sm text-stone-500">
            Set your preferred maximum budget in rupees.
          </p>

          <div className="mt-5 flex items-center rounded-xl border border-stone-200 bg-white px-5">
            <span className="text-stone-500">₹</span>

            <input
              type="number"
              min="1"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              className="w-full bg-transparent px-4 py-4 outline-none"
            />
          </div>

        </div>

        <div className="mt-12 flex flex-wrap gap-4">

          <button
            onClick={saveProfile}
            className="rounded-full bg-[#303329] px-8 py-4 text-sm font-semibold text-white transition hover:bg-[#4b4f40]"
          >
            Save my style
          </button>

          <Link
            to="/find-my-fit"
            className="flex items-center gap-2 rounded-full border border-stone-300 px-8 py-4 text-sm font-semibold transition hover:bg-white"
          >
            Find my fit
            <ArrowUpRight size={16} />
          </Link>

        </div>

      </section>

    </main>
  )
}