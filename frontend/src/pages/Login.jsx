import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(username, password)
      navigate('/')
    } catch (err) {
      setError(err?.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-app">
      <form onSubmit={onSubmit} className="card p-8 w-full max-w-sm">
        <div className="flex items-center gap-3 mb-6 justify-center">
          <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center text-xl">📷</div>
          <div>
            <div className="font-bold text-app-primary text-lg leading-tight">ANPR System</div>
            <div className="text-[10px] text-app-muted tracking-widest">SIGN IN</div>
          </div>
        </div>
        {error && <div className="bg-red-500/10 text-red-400 text-sm rounded-lg px-3 py-2 mb-4">{error}</div>}
        <label className="block text-xs text-app-secondary mb-1">Username</label>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full mb-4 px-3 py-2 rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
          autoFocus
        />
        <label className="block text-xs text-app-secondary mb-1">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full mb-6 px-3 py-2 rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
        />
        <button
          disabled={loading}
          className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary font-medium transition disabled:opacity-50"
        >
          {loading ? 'Signing in…' : 'Sign In'}
        </button>
        <p className="text-xs text-app-muted mt-4 text-center">
          Default: admin / admin123
        </p>
      </form>
    </div>
  )
}
