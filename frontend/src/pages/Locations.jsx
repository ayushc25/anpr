import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

export default function Locations() {
  const [locations, setLocations] = useState([])
  const [form, setForm] = useState({ name: '', description: '' })

  async function load() {
    const res = await client.get('/locations')
    setLocations(res.data)
  }

  useEffect(() => { load() }, [])

  async function submit(e) {
    e.preventDefault()
    await client.post('/locations', form)
    setForm({ name: '', description: '' })
    load()
  }

  async function remove(id) {
    if (!confirm('Delete this location?')) return
    await client.delete(`/locations/${id}`)
    load()
  }

  return (
    <Layout title="Locations">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <form onSubmit={submit} className="card p-4 space-y-3">
          <div className="font-semibold text-app-primary mb-1">Add Location</div>
          <input
            value={form.name}
            onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
            placeholder="Name"
            className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
          />
          <input
            value={form.description}
            onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
            placeholder="Description"
            className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
          />
          <button className="w-full py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm">Add</button>
        </form>
        <div className="card p-4 lg:col-span-2">
          <div className="font-semibold text-app-primary mb-3">Locations</div>
          <div className="space-y-2">
            {locations.map((l) => (
              <div key={l.id} className="flex items-center justify-between px-3 py-2 rounded-lg surface-1">
                <div>
                  <div className="text-app-primary text-sm">{l.name}</div>
                  <div className="text-xs text-app-muted">{l.description}</div>
                </div>
                <button onClick={() => remove(l.id)} className="text-red-400 hover:underline text-xs">Delete</button>
              </div>
            ))}
            {locations.length === 0 && <div className="text-app-muted text-sm">No locations added yet</div>}
          </div>
        </div>
      </div>
    </Layout>
  )
}
