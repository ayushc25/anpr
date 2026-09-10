import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

const PAGE_SIZE = 20

export default function Logs() {
  const [logs, setLogs] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    client
      .get('/logs', { params: { page, page_size: PAGE_SIZE } })
      .then((res) => {
        if (cancelled) return
        // Tolerate the pre-pagination shape (a bare array). An API that has
        // not been restarted after a deploy would otherwise hand us an
        // undefined `items`, and the .map() below throws inside render —
        // which unmounts the whole app, not just this page.
        const body = res.data
        const items = Array.isArray(body) ? body : body?.items ?? []
        setLogs(items)
        setTotal(Array.isArray(body) ? body.length : body?.total ?? 0)
      })
      .catch(() => {
        if (!cancelled) { setLogs([]); setTotal(0) }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    // A slow page that resolves after the user has already clicked on would
    // otherwise overwrite the newer one.
    return () => { cancelled = true }
  }, [page])

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <Layout title="Activity Logs">
      <div className="card p-4">
        <div className="font-semibold text-app-primary mb-3">Login Activity & Actions (last 3 months)</div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>User</th>
                <th>Action</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((l) => (
                <tr key={l.id}>
                  <td>{new Date(l.created_at).toLocaleString()}</td>
                  <td>{l.username}</td>
                  <td className="capitalize">{l.action.replace(/_/g, ' ')}</td>
                  <td>{l.details}</td>
                </tr>
              ))}
              {logs.length === 0 && !loading && (
                <tr><td colSpan={4} className="text-center text-app-muted py-8">No activity recorded yet</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between mt-4 text-sm text-app-secondary">
          <span>
            {total > 0
              ? `Page ${page} of ${totalPages} · ${total} entries`
              : 'No entries'}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              disabled={page <= 1 || loading}
              className="px-3 py-1.5 rounded-lg surface-1 hover-surface-2 disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
              disabled={page >= totalPages || loading}
              className="px-3 py-1.5 rounded-lg surface-1 hover-surface-2 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </Layout>
  )
}
