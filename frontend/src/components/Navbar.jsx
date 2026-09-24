
import { Link, NavLink } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `relative text-sm transition-colors duration-300 ${
      isActive
        ? 'font-semibold text-[#303329]'
        : 'text-stone-500 hover:text-[#303329]'
    }`

  return (
    <nav className="sticky top-0 z-50 border-b border-stone-200/70 bg-[#faf9f6]/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

        <Link
          to="/discover"
          className="text-2xl font-black tracking-tight"
        >
          fitmind<span className="text-[#858b6b]">.</span>
        </Link>

        <div className="hidden items-center gap-8 md:flex">
          <NavLink to="/discover" className={linkClass}>
            Discover
          </NavLink>

          <NavLink to="/style" className={linkClass}>
            Your style
          </NavLink>

          <NavLink to="/about" className={linkClass}>
            About
          </NavLink>
        </div>

        <Link
          to="/find-my-fit"
          className="group flex items-center gap-2 rounded-full bg-[#303329] px-5 py-3 text-sm font-semibold text-white transition-all duration-300 hover:bg-[#4b4f40]"
        >
          Find my fit

          <ArrowUpRight
            size={16}
            className="transition-transform duration-300 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
          />
        </Link>

      </div>
    </nav>
  )
}