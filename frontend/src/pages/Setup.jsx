import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import client from '../api/client'

export default function Setup() {
  const navigate = useNavigate()

  const [dbUrl, setDbUrl] = useState('')
  const [testingDb, setTestingDb] = useState(false)
  const [testResult, setTestResult] = useState(null)

  // Super Admin (Rising IT)
  const [superUsername, setSuperUsername] = useState('risingit')
  const [superFullName, setSuperFullName] = useState('Rising IT Super Admin')
  const [superPassword, setSuperPassword] = useState('')
  const [superPasswordConfirm, setSuperPasswordConfirm] = useState('')

  // Client Admin
  const [createClientAdmin, setCreateClientAdmin] = useState(true)
  const [clientUsername, setClientUsername] = useState('admin')
  const [clientFullName, setClientFullName] = useState('Client Administrator')
  const [clientPassword, setClientPassword] = useState('')
  const [clientPasswordConfirm, setClientPasswordConfirm] = useState('')

  const [loading, setLoading] = useState(false)
  const [statusLoading, setStatusLoading] = useState(true)
  const [error, setError] = useState('')
  const [successMsg, setSuccessMsg] = useState('')

  useEffect(() => {
    async function checkStatus() {
      try {
        const res = await client.get('/setup/status')
        if (res.data && !res.data.setup_required) {
          // Already initialized
          navigate('/login')
        }
      } catch (err) {
        console.warn('Could not check setup status', err)
      } finally {
        setStatusLoading(false)
      }
    }
    checkStatus()
  }, [navigate])

  async function onTestConnection() {
    if (!dbUrl.trim()) {
      setError('Please enter a PostgreSQL connection URL.')
      return
    }
    setError('')
    setTestResult(null)
    setTestingDb(true)
    try {
      const res = await client.post('/setup/test-db', { database_url: dbUrl.trim() })
      if (res.data.success) {
        setTestResult({ success: true, message: 'Connection successful! PostgreSQL database reached.' })
      } else {
        setTestResult({ success: false, message: res.data.error || 'Connection failed.' })
      }
    } catch (err) {
      setTestResult({
        success: false,
        message: err?.response?.data?.detail || err.message || 'Connection failed.',
      })
    } finally {
      setTestingDb(false)
    }
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setSuccessMsg('')

    if (!dbUrl.trim()) {
      setError('Database URL is required.')
      return
    }

    if (!superPassword || superPassword.length < 6) {
      setError('Super Admin password must be at least 6 characters.')
      return
    }

    if (superPassword !== superPasswordConfirm) {
      setError('Super Admin passwords do not match.')
      return
    }

    if (createClientAdmin) {
      if (!clientPassword || clientPassword.length < 6) {
        setError('Client Admin password must be at least 6 characters.')
        return
      }
      if (clientPassword !== clientPasswordConfirm) {
        setError('Client Admin passwords do not match.')
        return
      }
    }

    setLoading(true)
    try {
      const payload = {
        database_url: dbUrl.trim(),
        superadmin_username: superUsername.trim(),
        superadmin_password: superPassword,
        superadmin_full_name: superFullName.trim(),
        client_admin_username: createClientAdmin ? clientUsername.trim() : null,
        client_admin_password: createClientAdmin ? clientPassword : null,
        client_admin_full_name: createClientAdmin ? clientFullName.trim() : null,
      }

      const res = await client.post('/setup/initialize', payload)
      setSuccessMsg(res.data.message || 'Setup completed successfully! Redirecting to login...')
      setTimeout(() => {
        navigate('/login')
      }, 2000)
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to complete setup.')
    } finally {
      setLoading(false)
    }
  }

  if (statusLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-app text-app-muted">
        Checking system setup status...
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-app py-12 px-4 sm:px-6 lg:px-8">
      <div className="card p-8 w-full max-w-xl shadow-2xl border border-app">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6 pb-4 border-b border-app">
          <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center text-2xl shadow-lg">
            ⚙️
          </div>
          <div>
            <div className="font-extrabold text-app-primary text-xl tracking-tight">
              Rising IT ANPR Deployment
            </div>
            <div className="text-xs text-blue-400 font-semibold uppercase tracking-wider">
              System Initialization & Setup Wizard
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/15 border border-red-500/30 text-red-400 text-sm rounded-lg px-4 py-3 mb-6">
            <span className="font-semibold">Setup Error:</span> {error}
          </div>
        )}

        {successMsg && (
          <div className="bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-sm rounded-lg px-4 py-3 mb-6">
            <span className="font-semibold">Success:</span> {successMsg}
          </div>
        )}

        <form onSubmit={onSubmit} className="space-y-6">
          {/* Section 1: Database Connection */}
          <div className="bg-surface-alt p-5 rounded-xl border border-app space-y-3">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-semibold text-app-primary">
                1. PostgreSQL Database URL
              </label>
              <span className="text-[11px] text-app-muted">Client or local database</span>
            </div>
            <input
              type="text"
              value={dbUrl}
              onChange={(e) => {
                setDbUrl(e.target.value)
                setTestResult(null)
              }}
              placeholder="postgresql://username:password@localhost:5432/anpr_db"
              className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500 font-mono"
              required
            />
            <div className="flex items-center justify-between pt-1">
              <p className="text-xs text-app-muted">
                Format: <code className="text-xs text-blue-400">postgresql://user:pass@host:port/dbname</code>
              </p>
              <button
                type="button"
                onClick={onTestConnection}
                disabled={testingDb || !dbUrl.trim()}
                className="px-3 py-1.5 rounded-lg bg-surface hover:bg-hover-2 text-xs font-medium text-app-primary border border-app transition disabled:opacity-50"
              >
                {testingDb ? 'Testing Connection…' : 'Test Connection'}
              </button>
            </div>

            {testResult && (
              <div
                className={`text-xs rounded-lg px-3 py-2 mt-2 ${
                  testResult.success
                    ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                    : 'bg-red-500/15 text-red-300 border border-red-500/30'
                }`}
              >
                {testResult.message}
              </div>
            )}
          </div>

          {/* Section 2: Rising IT Super Admin */}
          <div className="bg-surface-alt p-5 rounded-xl border border-app space-y-3">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-semibold text-app-primary">
                2. Rising IT Super Admin (Master Account)
              </label>
              <span className="text-[11px] text-blue-400 font-medium">Full Permissions</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-app-secondary mb-1">Username</label>
                <input
                  type="text"
                  value={superUsername}
                  onChange={(e) => setSuperUsername(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-xs text-app-secondary mb-1">Display Name</label>
                <input
                  type="text"
                  value={superFullName}
                  onChange={(e) => setSuperFullName(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-xs text-app-secondary mb-1">Password</label>
                <input
                  type="password"
                  value={superPassword}
                  onChange={(e) => setSuperPassword(e.target.value)}
                  placeholder="Min 6 characters"
                  className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-xs text-app-secondary mb-1">Confirm Password</label>
                <input
                  type="password"
                  value={superPasswordConfirm}
                  onChange={(e) => setSuperPasswordConfirm(e.target.value)}
                  placeholder="Repeat password"
                  className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
            </div>
          </div>

          {/* Section 3: Client Admin (Optional / Handover) */}
          <div className="bg-surface-alt p-5 rounded-xl border border-app space-y-3">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={createClientAdmin}
                  onChange={(e) => setCreateClientAdmin(e.target.checked)}
                  className="rounded text-blue-600 focus:ring-blue-500 h-4 w-4"
                />
                <span className="text-sm font-semibold text-app-primary">
                  3. Create Client Admin Account
                </span>
              </label>
              <span className="text-[11px] text-app-muted">For handover</span>
            </div>

            {createClientAdmin && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="block text-xs text-app-secondary mb-1">Client Username</label>
                  <input
                    type="text"
                    value={clientUsername}
                    onChange={(e) => setClientUsername(e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs text-app-secondary mb-1">Client Display Name</label>
                  <input
                    type="text"
                    value={clientFullName}
                    onChange={(e) => setClientFullName(e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs text-app-secondary mb-1">Client Password</label>
                  <input
                    type="password"
                    value={clientPassword}
                    onChange={(e) => setClientPassword(e.target.value)}
                    placeholder="Min 6 characters"
                    className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs text-app-secondary mb-1">Confirm Client Password</label>
                  <input
                    type="password"
                    value={clientPasswordConfirm}
                    onChange={(e) => setClientPasswordConfirm(e.target.value)}
                    placeholder="Repeat password"
                    className="w-full px-3 py-2 text-sm rounded-lg bg-app border border-app text-app-primary focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold shadow-lg shadow-blue-500/20 transition disabled:opacity-50 text-sm"
          >
            {loading ? 'Initializing Database & Migrating Schema…' : 'Initialize & Deploy System'}
          </button>
        </form>
      </div>
    </div>
  )
}

