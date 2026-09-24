
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="bg-[#303329] text-white">
      <div className="mx-auto max-w-7xl px-6 py-12">

        <div className="flex flex-col justify-between gap-8 md:flex-row md:items-center">

          <div>
            <Link
              to="/discover"
              className="text-3xl font-black tracking-tight"
            >
              fitmind<span className="text-[#b9c09c]">.</span>
            </Link>

            <p className="mt-3 max-w-sm text-sm leading-6 text-stone-400">
              Personal style, powered by intelligence.
              Fashion that feels like you.
            </p>
          </div>

          <div className="flex flex-wrap gap-6 text-sm text-stone-300">
            <Link to="/discover" className="hover:text-white">
              Discover
            </Link>

            <Link to="/style" className="hover:text-white">
              Your style
            </Link>

            <Link to="/about" className="hover:text-white">
              About
            </Link>

            <Link to="/find-my-fit" className="hover:text-white">
              Find my fit
            </Link>
          </div>

        </div>

        <div className="mt-10 border-t border-white/10 pt-6">
          <p className="text-xs text-stone-500">
            © {new Date().getFullYear()} FitMind AI. Built with care.
          </p>
        </div>

      </div>
    </footer>
  )
}