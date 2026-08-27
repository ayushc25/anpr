import { useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'
import StatusBadge from '../components/StatusBadge'

export default function VehicleSearch() {
  const [q, setQ] = useState('')
  const [vehicles, setVehicles] = useState([])
  const [selected, setSelected] = useState(null)
  const [history, setHistory] = useState([])

  async function search(e) {
    e?.preventDefault()
    const res = await client.get('/vehicles', { params: { q } })
    setVehicles(res.data)
  }

  async function select(v) {
    setSelected(v)
    const res = await client.get(`/vehicles/${v.id}/history`)
    setHistory(res.data)
  }

  return (
    <Layout title="Vehicle Search">
      <form onSubmit={search} className="flex gap-3 mb-4">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search by plate number or owner name..."
          className="flex-1 px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
        />
        <button className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm">Search</button>
      </form>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card p-4 lg:col-span-1">
          <div className="font-semibold text-app-primary mb-3">Results</div>
          <div className="space-y-2 max-h-[520px] overflow-y-auto">
            {vehicles.map((v) => (
              <button
                key={v.id}
                onClick={() => select(v)}
                className={`w-full text-left px-3 py-2 rounded-lg transition ${
                  selected?.id === v.id ? 'bg-blue-600/20 border border-blue-500/40' : 'surface-1 hover-surface-2'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-mono text-app-primary">{v.plate_number}</span>
                  <StatusBadge status={v.status} />
                </div>
                <div className="text-xs text-app-secondary mt-1">
                  {v.owner_name || 'No owner recorded'}{v.flat_number ? ` · Flat ${v.flat_number}` : ''} · {v.vehicle_type}
                </div>
              </button>
            ))}
            {vehicles.length === 0 && <div className="text-app-muted text-sm">No results yet</div>}
          </div>
        </div>

        <div className="card p-4 lg:col-span-2">
          <div className="font-semibold text-app-primary mb-3">Vehicle History</div>
          {!selected && <div className="text-app-muted text-sm">Select a vehicle to view its detection history</div>}
          {selected && (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Camera</th>
                    <th>Direction</th>
                    <th>Confidence</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h) => (
                    <tr key={h.id}>
                      <td>{new Date(h.detected_at).toLocaleString()}</td>
                      <td>{h.camera_name || '—'}</td>
                      <td className="uppercase">{h.direction}</td>
                      <td>{Math.round(h.confidence * 100)}%</td>
                      <td><StatusBadge status={h.status} /></td>
                    </tr>
                  ))}
                  {history.length === 0 && (
                    <tr><td colSpan={5} className="text-center text-app-muted py-6">No history for this vehicle</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </Layout>
  )
}
