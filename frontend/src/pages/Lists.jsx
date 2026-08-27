import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'
import StatusBadge from '../components/StatusBadge'

const EMPTY = { plate_number: '', owner_name: '', flat_number: '', vehicle_type: 'car', status: 'whitelist', notes: '' }

export default function Lists() {
  const [vehicles, setVehicles] = useState([])
  const [filter, setFilter] = useState('')
  const [form, setForm] = useState(EMPTY)
  const [editingId, setEditingId] = useState(null)
  const [error, setError] = useState('')

  async function load() {
    const res = await client.get('/vehicles', { params: filter ? { status: filter } : {} })
    setVehicles(res.data)
  }

  useEffect(() => { load() }, [filter])

  async function submit(e) {
    e.preventDefault()
    setError('')
    try {
      if (editingId) {
        await client.put(`/vehicles/${editingId}`, form)
      } else {
        await client.post('/vehicles', form)
      }
      setForm(EMPTY)
      setEditingId(null)
      load()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to save')
    }
  }

  function edit(v) {
    setEditingId(v.id)
    setForm({
      plate_number: v.plate_number,
      owner_name: v.owner_name,
      flat_number: v.flat_number || '',
      vehicle_type: v.vehicle_type,
      status: v.status,
      notes: v.notes || '',
    })
  }

  async function remove(id) {
    if (!confirm('Delete this vehicle record?')) return
    await client.delete(`/vehicles/${id}`)
    load()
  }

  return (
    <Layout title="Blacklist / Whitelist">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <form onSubmit={submit} className="card p-4 lg:col-span-1 space-y-3">
          <div className="font-semibold text-app-primary mb-1">{editingId ? 'Edit Vehicle' : 'Add Vehicle'}</div>
          {error && <div className="bg-red-500/10 text-red-400 text-xs rounded px-2 py-1">{error}</div>}
          <div>
            <label className="block text-xs text-app-secondary mb-1">Plate Number</label>
            <input
              value={form.plate_number}
              disabled={!!editingId}
              onChange={(e) => setForm((f) => ({ ...f, plate_number: e.target.value.toUpperCase() }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm disabled:opacity-50"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-app-secondary mb-1">Owner Name</label>
              <input
                value={form.owner_name}
                onChange={(e) => setForm((f) => ({ ...f, owner_name: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-app-secondary mb-1">Flat Number</label>
              <input
                value={form.flat_number}
                onChange={(e) => setForm((f) => ({ ...f, flat_number: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-app-secondary mb-1">Vehicle Type</label>
              <select
                value={form.vehicle_type}
                onChange={(e) => setForm((f) => ({ ...f, vehicle_type: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              >
                <option value="car">Car</option>
                <option value="motorbike">Motorbike</option>
                <option value="bicycle">Bicycle</option>
                <option value="truck">Truck</option>
                <option value="bus">Bus</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-app-secondary mb-1">Status</label>
              <select
                value={form.status}
                onChange={(e) => setForm((f) => ({ ...f, status: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              >
                <option value="registered">Registered</option>
                <option value="whitelist">Whitelist</option>
                <option value="blacklist">Blacklist</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">Notes</label>
            <textarea
              value={form.notes}
              onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              rows={2}
            />
          </div>
          <div className="flex gap-2">
            <button className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm">
              {editingId ? 'Update' : 'Add'}
            </button>
            {editingId && (
              <button
                type="button"
                onClick={() => { setEditingId(null); setForm(EMPTY) }}
                className="px-3 py-2 rounded-lg surface-1 hover-surface-2 text-app-secondary text-sm"
              >
                Cancel
              </button>
            )}
          </div>
        </form>

        <div className="card p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <div className="font-semibold text-app-primary">Vehicles</div>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-app border border-app text-app-primary text-sm"
            >
              <option value="">All</option>
              <option value="registered">Registered</option>
              <option value="whitelist">Whitelist</option>
              <option value="blacklist">Blacklist</option>
            </select>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Plate</th>
                  <th>Owner</th>
                  <th>Flat No.</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map((v) => (
                  <tr key={v.id}>
                    <td className="font-mono">{v.plate_number}</td>
                    <td>{v.owner_name || '—'}</td>
                    <td>{v.flat_number || '—'}</td>
                    <td className="capitalize">{v.vehicle_type}</td>
                    <td><StatusBadge status={v.status} /></td>
                    <td className="text-right space-x-2 whitespace-nowrap">
                      <button onClick={() => edit(v)} className="text-blue-400 hover:underline text-xs">Edit</button>
                      <button onClick={() => remove(v.id)} className="text-red-400 hover:underline text-xs">Delete</button>
                    </td>
                  </tr>
                ))}
                {vehicles.length === 0 && (
                  <tr><td colSpan={6} className="text-center text-app-muted py-8">No vehicles found</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  )
}
