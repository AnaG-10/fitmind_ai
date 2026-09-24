
import { Link } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'
import { motion } from 'framer-motion'

export default function About() {
  return (
    <main className="bg-[#faf9f6]">

      {/* INTRO */}

      <section className="mx-auto max-w-7xl px-6 py-24 md:py-36">

        <motion.div
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >

          <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
            The FitMind approach
          </p>

          <h1 className="mt-8 max-w-5xl text-5xl font-medium leading-tight tracking-tight md:text-8xl">
            Fashion meets
            <br />
            <span className="font-serif italic text-[#858b6b]">
              intelligence.
            </span>
          </h1>

          <p className="mt-8 max-w-2xl text-lg leading-8 text-stone-600">
            FitMind AI brings together personal style preferences,
            semantic product discovery, and AI-generated styling
            suggestions to make fashion discovery more personal.
          </p>

        </motion.div>

      </section>

      {/* IMAGE */}

      <section className="px-4 md:px-8">

        <div className="relative h-[350px] overflow-hidden md:h-[600px]">

          <img
            src="https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=2200&q=90"
            alt="Fashion and personal style"
            className="h-full w-full object-cover"
          />

          <div className="absolute inset-0 bg-black/20" />

          <div className="absolute bottom-8 left-8 text-white md:bottom-16 md:left-16">
            <p className="text-xs font-semibold uppercase tracking-[0.25em]">
              FitMind AI
            </p>

            <h2 className="mt-4 font-serif text-4xl italic md:text-6xl">
              Style, made personal.
            </h2>
          </div>

        </div>

      </section>

      {/* HOW IT WORKS */}

      <section className="mx-auto max-w-7xl px-6 py-24 md:py-32">

        <div className="grid gap-16 md:grid-cols-2">

          <div>
            <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
              How it works
            </p>

            <h2 className="mt-6 text-4xl font-medium leading-tight md:text-6xl">
              A little intelligence.
              <br />
              <span className="font-serif italic text-[#858b6b]">
                A lot of personal style.
              </span>
            </h2>
          </div>

          <div className="space-y-10">

            <div className="border-b border-stone-200 pb-8">
              <p className="text-xs font-semibold tracking-widest text-[#858b6b]">
                01
              </p>

              <h3 className="mt-3 text-2xl font-medium">
                Tell us about you
              </h3>

              <p className="mt-3 leading-7 text-stone-600">
                Choose your preferences, including occasion,
                clothing category, fit, budget, and colours.
              </p>
            </div>

            <div className="border-b border-stone-200 pb-8">
              <p className="text-xs font-semibold tracking-widest text-[#858b6b]">
                02
              </p>

              <h3 className="mt-3 text-2xl font-medium">
                Discover relevant pieces
              </h3>

              <p className="mt-3 leading-7 text-stone-600">
                FitMind uses product filtering and semantic
                similarity search to retrieve matching items
                from its fashion catalogue.
              </p>
            </div>

            <div>
              <p className="text-xs font-semibold tracking-widest text-[#858b6b]">
                03
              </p>

              <h3 className="mt-3 text-2xl font-medium">
                Get your styling suggestions
              </h3>

              <p className="mt-3 leading-7 text-stone-600">
                An AI stylist uses the retrieved products to
                generate personalised recommendations and
                additional styling ideas.
              </p>
            </div>

          </div>

        </div>

      </section>

      {/* CTA */}

      <section className="bg-[#e9ebdf] px-6 py-24 text-center md:py-32">

        <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#777e5b]">
          Your style journey starts here
        </p>

        <h2 className="mx-auto mt-6 max-w-3xl text-5xl font-medium md:text-7xl">
          Ready to find
          <br />
          <span className="font-serif italic text-[#777e5b]">
            your fit?
          </span>
        </h2>

        <Link
          to="/find-my-fit"
          className="mt-10 inline-flex items-center gap-3 rounded-full bg-[#303329] px-8 py-4 text-sm font-semibold text-white transition hover:bg-[#4b4f40]"
        >
          Find my fit
          <ArrowUpRight size={17} />
        </Link>

      </section>

    </main>
  )
}