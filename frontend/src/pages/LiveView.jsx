import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

export default function LiveView() {
  const [cameras, setCameras] = useState([])

  useEffect(() => {
    client.get('/cameras').then((res) => setCameras(res.data))
  }, [])

  return (
    <Layout title="Live View">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {cameras.map((c) => (
          <div key={c.id} className="card p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-app-primary font-medium">{c.name}</span>
              <span className={`text-xs font-semibold ${c.is_online ? 'text-emerald-400' : 'text-red-400'}`}>
                {c.is_online ? '● LIVE' : '● OFFLINE'}
              </span>
            </div>
            <div className="rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center">
              <img
                src={`/api/cameras/${c.id}/stream`}
                alt={c.name}
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        ))}
        {cameras.length === 0 && (
          <div className="text-app-muted col-span-2 text-center py-12">
            No cameras configured. Add one from the Cameras page.
          </div>
        )}
      </div>
    </Layout>
  )
}
