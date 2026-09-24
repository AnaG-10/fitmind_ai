
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'

import Navbar from './components/Navbar'
import Footer from './components/Footer'

import Discover from './pages/Discover'
import YourStyle from './pages/Your-Style'
import About from './pages/About'
import FindMyFit from './pages/FindMyFit'

function AnimatedRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={<Navigate to="/discover" replace />}
        />

        <Route
          path="/discover"
          element={<Discover />}
        />

        <Route
          path="/style"
          element={<YourStyle />}
        />

        <Route
          path="/about"
          element={<About />}
        />

        <Route
          path="/find-my-fit"
          element={<FindMyFit />}
        />

        <Route
          path="*"
          element={<Navigate to="/discover" replace />}
        />
      </Routes>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#faf9f6] text-[#25251f]">
        <Navbar />

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <AnimatedRoutes />
        </motion.div>

        <Footer />
      </div>
    </BrowserRouter>
  )
}