import { useEffect, useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import client from '../api/client'
import Layout from '../components/Layout'
import StatCard from '../components/StatCard'
import StatusBadge from '../components/StatusBadge'
import { useTheme } from '../context/ThemeContext'

export default function Dashboard() {
  const { theme } = useTheme()
  const [stats, setStats] = useState(null)
  const [trend, setTrend] = useState([])
  const [latest, setLatest] = useState([])
  const [cameras, setCameras] = useState([])
  const tooltipStyle = theme === 'light'
    ? { background: '#ffffff', border: '1px solid #dde1ea' }
    : { background: '#12172a', border: '1px solid #232a3d' }

  async function load() {
    const [s, t, l, c] = await Promise.all([
      client.get('/dashboard/stats'),
      client.get('/dashboard/trend'),
      client.get('/dashboard/latest'),
      client.get('/cameras'),
    ])
    setStats(s.data)
    setTrend(t.data)
    setLatest(l.data)
    setCameras(c.data)
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [])

  const pieData = stats ? [
    { name: 'Entry', value: stats.entries_today || 0 },
    { name: 'Exit', value: stats.exits_today || 0 },
  ] : []

  return (
    <Layout title="Dashboard">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        <StatCard icon="🚗" label="Total Vehicles Today" value={stats?.total_vehicles_today ?? '—'} color="blue" />
        <StatCard icon="🅿️" label="Vehicles Inside" value={stats?.vehicles_inside ?? '—'} color="green" sub="Live inside premises" />
        <StatCard icon="🚦" label="Vehicles Out" value={stats?.exits_today ?? '—'} color="blue" sub="Exited today" />
        <StatCard icon="⚠️" label="Blacklisted Vehicles" value={stats?.blacklisted_count ?? '—'} color="red" />
        <StatCard icon="❓" label="Unknown Vehicles Today" value={stats?.unknown_today ?? '—'} color="yellow" />
        <StatCard
          icon="🎥"
          label="Active Cameras"
          value={stats ? `${stats.active_cameras} / ${stats.total_cameras}` : '—'}
          color="purple"
          sub="Online"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <div className="font-semibold text-app-primary mb-3">Latest ANPR Detections</div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Plate</th>
                  <th>Type</th>
                  <th>Time</th>
                  <th>Camera</th>
                  <th>Confidence</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {latest.map((e) => (
                  <tr key={e.id}>
                    <td className="font-mono">{e.plate_number}</td>
                    <td className="capitalize">{e.vehicle_type}</td>
                    <td>{new Date(e.detected_at).toLocaleTimeString()}</td>
                    <td>{e.camera_name || '—'}</td>
                    <td>{Math.round(e.confidence * 100)}%</td>
                    <td><StatusBadge status={e.status} /></td>
                  </tr>
                ))}
                {latest.length === 0 && (
                  <tr><td colSpan={6} className="text-center text-app-muted py-6">No detections yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card p-4">
          <div className="font-semibold text-app-primary mb-3">Camera Status</div>
          <div className="space-y-2">
            {cameras.map((c) => (
              <div key={c.id} className="flex items-center justify-between px-3 py-2 rounded-lg surface-1">
                <span className="text-app-secondary text-sm">{c.name}</span>
                <span className={`text-xs font-semibold ${c.is_online ? 'text-emerald-400' : 'text-red-400'}`}>
                  {c.is_online ? 'Online' : 'Offline'}
                </span>
              </div>
            ))}
            {cameras.length === 0 && <div className="text-app-muted text-sm">No cameras configured yet</div>}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card p-4 lg:col-span-2">
          <div className="font-semibold text-app-primary mb-3">Vehicle Trend (Today)</div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={trend}>
              <defs>
                <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="hour" tick={{ fill: '#7c8398', fontSize: 11 }} interval={2} />
              <YAxis tick={{ fill: '#7c8398', fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="url(#colorCount)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-4">
          <div className="font-semibold text-app-primary mb-3">Entry / Exit Summary</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                <Cell fill="#3b82f6" />
                <Cell fill="#2ecc71" />
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Layout>
  )
}
