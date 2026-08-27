import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'

export default function Topbar({ title }) {
  const { logout } = useAuth()
  const { theme, toggleTheme } = useTheme()
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-app-soft bg-app-alt-blur sticky top-0 z-10">
      <h1 className="text-lg font-semibold text-app-primary">{title}</h1>
      <div className="flex items-center gap-5 text-sm text-app-secondary">
        <span>{now.toLocaleDateString()}</span>
        <span>{now.toLocaleTimeString()}</span>
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          className="px-2.5 py-1.5 rounded-lg surface-1 hover-surface-2 text-app-secondary transition"
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
        <button
          onClick={logout}
          className="px-3 py-1.5 rounded-lg surface-1 hover-surface-2 text-app-secondary transition"
        >
          Logout
        </button>
      </div>
    </header>
  )
}
