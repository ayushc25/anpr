import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'
import StatusBadge from '../components/StatusBadge'
import ImageLightbox from '../components/ImageLightbox'

const PAGE_SIZE = 20

const COLOR_SWATCH = {
  white: '#f5f5f5', black: '#1a1a1a', gray: '#8a8a8a', silver: '#c4c4c4',
  red: '#e03131', orange: '#f08c00', yellow: '#f5d90a', green: '#2f9e44',
  teal: '#0ca678', blue: '#1971c2', purple: '#7048e8', pink: '#e64980',
}

function ColorTag({ color }) {
  if (!color) return <span className="text-app-muted">—</span>
  return (
    <span className="inline-flex items-center gap-1.5 capitalize">
      <span
        className="inline-block w-2.5 h-2.5 rounded-full border border-white/20"
        style={{ background: COLOR_SWATCH[color] || '#666' }}
      />
      {color}
    </span>
  )
}

export default function Events() {
  const [events, setEvents] = useState([])
  const [filters, setFilters] = useState({ plate: '', status: '' })
  const [preview, setPreview] = useState(null)
  const [page, setPage] = useState(0)
  const [hasNext, setHasNext] = useState(false)

  async function load() {
    const params = { limit: PAGE_SIZE + 1, offset: page * PAGE_SIZE }
    if (filters.plate) params.plate = filters.plate
    if (filters.status) params.status = filters.status
    const res = await client.get('/events', { params })
    setHasNext(res.data.length > PAGE_SIZE)
    setEvents(res.data.slice(0, PAGE_SIZE))
  }

  useEffect(() => {
    setPage(0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.plate, filters.status])

  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, page])

  return (
    <Layout title="ANPR Events">
      <div className="card p-4">
        <div className="flex gap-3 mb-4">
          <input
            placeholder="Search plate number..."
            value={filters.plate}
            onChange={(e) => setFilters((f) => ({ ...f, plate: e.target.value }))}
            className="px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
          />
          <select
            value={filters.status}
            onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}
            className="px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
          >
            <option value="">All Status</option>
            <option value="registered">Registered</option>
            <option value="whitelist">Whitelist</option>
            <option value="blacklist">Blacklist</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Image</th>
                <th>Plate</th>
                <th>Type</th>
                <th>Vehicle Color</th>
                <th>Plate Color</th>
                <th>Time</th>
                <th>Camera</th>
                <th>Direction</th>
                <th>Owner</th>
                <th>Confidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id}>
                  <td>
                    {e.image_path ? (
                      <img
                        src={`/api/media/${e.image_path}`}
                        alt=""
                        className="w-14 h-10 object-cover rounded cursor-pointer hover:opacity-80"
                        onClick={() => setPreview(e)}
                      />
                    ) : (
                      <div className="w-14 h-10 surface-1 rounded" />
                    )}
                  </td>
                  <td className="font-mono">{e.plate_number}</td>
                  <td className="capitalize">{e.vehicle_type}</td>
                  <td><ColorTag color={e.vehicle_color} /></td>
                  <td><ColorTag color={e.plate_color} /></td>
                  <td>{new Date(e.detected_at).toLocaleString()}</td>
                  <td>{e.camera_name || '—'}</td>
                  <td className="uppercase">{e.direction}</td>
                  <td>{e.owner_name || '—'}</td>
                  <td>{Math.round(e.confidence * 100)}%</td>
                  <td><StatusBadge status={e.status} /></td>
                </tr>
              ))}
              {events.length === 0 && (
                <tr><td colSpan={11} className="text-center text-app-muted py-8">No events found</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-4">
          <span className="text-sm text-app-muted">Page {page + 1}</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(p - 1, 0))}
              disabled={page === 0}
              className="px-3 py-1.5 rounded-lg surface-1 text-sm text-app-primary disabled:opacity-40 disabled:cursor-not-allowed hover-surface-2"
            >
              Previous
            </button>
            <button
              onClick={() => setPage((p) => (hasNext ? p + 1 : p))}
              disabled={!hasNext}
              className="px-3 py-1.5 rounded-lg surface-1 text-sm text-app-primary disabled:opacity-40 disabled:cursor-not-allowed hover-surface-2"
            >
              Next
            </button>
          </div>
        </div>
      </div>
      {preview && (
        <ImageLightbox
          src={`/api/media/${preview.image_path}`}
          alt={preview.plate_number}
          filename={`${preview.plate_number}_${preview.id}.jpg`}
          onClose={() => setPreview(null)}
        />
      )}
    </Layout>
  )
}
