import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

export default function Logs() {
  const [logs, setLogs] = useState([])

  useEffect(() => {
    client.get('/logs').then((res) => setLogs(res.data))
  }, [])

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
              {logs.length === 0 && (
                <tr><td colSpan={4} className="text-center text-app-muted py-8">No activity recorded yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  )
}
