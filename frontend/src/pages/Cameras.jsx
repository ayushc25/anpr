import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

const EMPTY = { name: '', rtsp_url: '', location_id: '', direction: 'both', is_active: true }

export default function Cameras() {
  const [cameras, setCameras] = useState([])
  const [locations, setLocations] = useState([])
  const [form, setForm] = useState(EMPTY)
  const [editingId, setEditingId] = useState(null)
  const [error, setError] = useState('')

  async function load() {
    const [c, l] = await Promise.all([client.get('/cameras'), client.get('/locations')])
    setCameras(c.data)
    setLocations(l.data)
  }

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [])

  async function submit(e) {
    e.preventDefault()
    setError('')
    const payload = { ...form, location_id: form.location_id ? Number(form.location_id) : null }
    try {
      if (editingId) {
        await client.put(`/cameras/${editingId}`, payload)
      } else {
        await client.post('/cameras', payload)
      }
      setForm(EMPTY)
      setEditingId(null)
      load()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to save camera')
    }
  }

  function edit(c) {
    setEditingId(c.id)
    setForm({ name: c.name, rtsp_url: c.rtsp_url, location_id: c.location_id || '', direction: c.direction, is_active: c.is_active })
  }

  async function remove(id) {
    if (!confirm('Delete this camera?')) return
    await client.delete(`/cameras/${id}`)
    load()
  }

  return (
    <Layout title="Cameras">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <form onSubmit={submit} className="card p-4 space-y-3">
          <div className="font-semibold text-app-primary mb-1">{editingId ? 'Edit Camera' : 'Add Camera'}</div>
          {error && <div className="bg-red-500/10 text-red-400 text-xs rounded px-2 py-1">{error}</div>}
          <div>
            <label className="block text-xs text-app-secondary mb-1">Name</label>
            <input
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">RTSP URL</label>
            <input
              value={form.rtsp_url}
              onChange={(e) => setForm((f) => ({ ...f, rtsp_url: e.target.value }))}
              placeholder="rtsp://user:pass@ip:554/stream"
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-app-secondary mb-1">Location</label>
              <select
                value={form.location_id}
                onChange={(e) => setForm((f) => ({ ...f, location_id: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              >
                <option value="">—</option>
                {locations.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-app-secondary mb-1">Direction</label>
              <select
                value={form.direction}
                onChange={(e) => setForm((f) => ({ ...f, direction: e.target.value }))}
                className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
              >
                <option value="in">IN</option>
                <option value="out">OUT</option>
                <option value="both">Both</option>
              </select>
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-app-secondary">
            <input type="checkbox" checked={form.is_active} onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.checked }))} />
            Active
          </label>
          <div className="flex gap-2">
            <button className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm">
              {editingId ? 'Update' : 'Add'}
            </button>
            {editingId && (
              <button type="button" onClick={() => { setEditingId(null); setForm(EMPTY) }} className="px-3 py-2 rounded-lg surface-1 text-app-secondary text-sm">
                Cancel
              </button>
            )}
          </div>
        </form>

        <div className="card p-4 lg:col-span-2">
          <div className="font-semibold text-app-primary mb-3">Configured Cameras</div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Location</th>
                  <th>Direction</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {cameras.map((c) => (
                  <tr key={c.id}>
                    <td>{c.name}</td>
                    <td>{locations.find((l) => l.id === c.location_id)?.name || '—'}</td>
                    <td className="uppercase">{c.direction}</td>
                    <td>
                      <span className={`text-xs font-semibold ${c.is_online ? 'text-emerald-400' : 'text-red-400'}`}>
                        {c.is_online ? 'Online' : 'Offline'}
                      </span>
                    </td>
                    <td className="text-right space-x-2 whitespace-nowrap">
                      <button onClick={() => edit(c)} className="text-blue-400 hover:underline text-xs">Edit</button>
                      <button onClick={() => remove(c.id)} className="text-red-400 hover:underline text-xs">Delete</button>
                    </td>
                  </tr>
                ))}
                {cameras.length === 0 && (
                  <tr><td colSpan={5} className="text-center text-app-muted py-8">No cameras added yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  )
}
