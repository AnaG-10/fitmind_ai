
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowDown, ArrowUpRight } from 'lucide-react'

const collections = [
  {
    title: 'Everyday essentials',
    subtitle: 'Pieces for your everyday wardrobe',
    image:
      'https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=1000&q=85',
    category: 'Casual',
  },
  {
    title: 'The modern classic',
    subtitle: 'Tailored, timeless, effortlessly you',
    image:
      'https://images.unsplash.com/photo-1487222477894-8943e31ef7b2?auto=format&fit=crop&w=1000&q=85',
    category: 'Formal',
  },
  {
    title: 'A little statement',
    subtitle: 'For the moments that deserve more',
    image:
      'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=1000&q=85',
    category: 'Party',
  },
]

export default function Discover() {
  return (
    <main className="overflow-hidden">

      {/* HERO */}

      <section className="relative min-h-[650px] bg-[#e8e8dd] md:min-h-[780px]">

        <img
          src="https://images.unsplash.com/photo-1539109136881-3be0616acf4b?auto=format&fit=crop&w=2200&q=90"
          alt="Fashion editorial"
          className="absolute inset-0 h-full w-full object-cover"
        />

        <div className="absolute inset-0 bg-gradient-to-r from-black/55 via-black/20 to-transparent" />

        <div className="relative mx-auto flex min-h-[650px] max-w-7xl flex-col justify-center px-6 py-24 md:min-h-[780px]">

          <motion.div
            initial={{ opacity: 0, y: 35 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9 }}
            className="max-w-3xl text-white"
          >
            <p className="mb-8 text-xs font-semibold uppercase tracking-[0.3em] text-white/80">
              Your personal AI stylist
            </p>

            <h1 className="text-6xl font-medium leading-[1.02] tracking-tight md:text-8xl lg:text-9xl">
              Fashion that
              <br />
              <span className="font-serif italic text-[#dce0c9]">
                feels like you.
              </span>
            </h1>

            <p className="mt-8 max-w-lg text-base leading-8 text-white/85 md:text-lg">
              Discover pieces that fit your body, your budget,
              and your personality. Your wardrobe, reimagined
              around you.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">

              <Link
                to="/find-my-fit"
                className="group flex items-center gap-3 rounded-full bg-[#faf9f6] px-7 py-4 text-sm font-semibold text-[#303329] transition hover:bg-[#dce0c9]"
              >
                Find my fit
                <ArrowUpRight
                  size={17}
                  className="transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                />
              </Link>

              <Link
                to="/style"
                className="rounded-full border border-white/50 px-7 py-4 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Build my style profile
              </Link>

            </div>
          </motion.div>

          <div className="absolute bottom-8 left-6 flex items-center gap-3 text-xs uppercase tracking-widest text-white/70">
            <ArrowDown size={16} />
            Explore your style
          </div>

          <p className="absolute bottom-8 right-6 hidden text-xs tracking-widest text-white/70 md:block">
            FITMIND AI · PERSONAL STYLE
          </p>

        </div>
      </section>

      {/* INTRODUCTION */}

      <section className="mx-auto grid max-w-7xl gap-10 px-6 py-24 md:grid-cols-2 md:items-end md:py-32">

        <div>
          <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
            Style, reimagined
          </p>

          <h2 className="mt-6 text-4xl font-medium leading-tight tracking-tight md:text-6xl">
            Less scrolling.
            <br />
            <span className="font-serif italic text-[#858b6b]">
              More wearing.
            </span>
          </h2>
        </div>

        <div>
          <p className="max-w-lg text-base leading-8 text-stone-600 md:text-lg">
            Your style is personal. Your recommendations should be too.
            FitMind combines your preferences with intelligent product
            matching to help you discover pieces that feel right for you.
          </p>

          <Link
            to="/about"
            className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-[#62694c]"
          >
            Discover how it works
            <ArrowUpRight size={16} />
          </Link>
        </div>

      </section>

      {/* COLLECTIONS */}

      <section className="bg-[#f0f0e8] py-24 md:py-32">

        <div className="mx-auto max-w-7xl px-6">

          <div className="mb-12 flex flex-wrap items-end justify-between gap-6">

            <div>
              <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
                The edit
              </p>

              <h2 className="mt-4 text-4xl font-medium md:text-5xl">
                Find your <span className="font-serif italic">feeling.</span>
              </h2>
            </div>

            <Link
              to="/find-my-fit"
              className="flex items-center gap-2 text-sm font-semibold"
            >
              Explore recommendations
              <ArrowUpRight size={16} />
            </Link>

          </div>

          <div className="grid gap-6 md:grid-cols-3">

            {collections.map((item, index) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 25 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.15 }}
              >
                <Link
                  to="/find-my-fit"
                  className="group block"
                >

                  <div className="relative aspect-[4/5] overflow-hidden bg-[#dedfd3]">

                    <img
                      src={item.image}
                      alt={item.title}
                      loading="lazy"
                      className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-105"
                    />

                    <span className="absolute left-5 top-5 rounded-full bg-[#faf9f6]/90 px-4 py-2 text-xs font-semibold">
                      {item.category}
                    </span>

                    <span className="absolute bottom-5 right-5 flex h-11 w-11 items-center justify-center rounded-full bg-[#faf9f6] transition-transform duration-300 group-hover:rotate-45">
                      <ArrowUpRight size={20} />
                    </span>

                  </div>

                  <h3 className="mt-5 text-xl font-medium">
                    {item.title}
                  </h3>

                  <p className="mt-2 text-sm text-stone-500">
                    {item.subtitle}
                  </p>

                </Link>
              </motion.div>
            ))}

          </div>

        </div>
      </section>

      {/* CTA */}

      <section className="mx-auto max-w-7xl px-6 py-24 text-center md:py-32">

        <p className="text-xs font-bold uppercase tracking-[0.25em] text-[#858b6b]">
          Your wardrobe starts here
        </p>

        <h2 className="mx-auto mt-6 max-w-3xl text-5xl font-medium leading-tight md:text-7xl">
          Your style.
          <br />
          <span className="font-serif italic text-[#858b6b]">
            Your rules.
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