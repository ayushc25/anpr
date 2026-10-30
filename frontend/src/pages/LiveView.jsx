import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'
import StatusBadge from '../components/StatusBadge'
import ImageLightbox from '../components/ImageLightbox'

export default function LiveView() {
  const [cameras, setCameras] = useState([])
  const [liveEvents, setLiveEvents] = useState([])
  const [preview, setPreview] = useState(null)

  useEffect(() => {
    client.get('/cameras').then((res) => setCameras(res.data))
  }, [])

  useEffect(() => {
    async function fetchLiveEvents() {
      try {
        const res = await client.get('/events', { params: { limit: 10 } })
        setLiveEvents(res.data)
      } catch (err) {
        console.warn('Could not fetch live events', err)
      }
    }

    fetchLiveEvents()
    const timer = setInterval(fetchLiveEvents, 3000)
    return () => clearInterval(timer)
  }, [])

  return (
    <Layout title="Live View & Gate Feed">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Camera Video Streams (2 Columns on large screens) */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-app-primary uppercase tracking-wider flex items-center gap-2">
              <span>📹</span> Live Camera Feeds
            </h2>
            <span className="text-xs text-app-muted">
              {cameras.filter((c) => c.is_online).length} of {cameras.length} cameras active
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {cameras.map((c) => (
              <div key={c.id} className="card p-3 shadow-sm border border-app">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-app-primary font-medium text-sm">{c.name}</span>
                  <span
                    className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                      c.is_online
                        ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                        : 'bg-red-500/15 text-red-400 border border-red-500/30'
                    }`}
                  >
                    {c.is_online ? '● LIVE' : '● OFFLINE'}
                  </span>
                </div>
                <div className="rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center border border-app/50 shadow-inner">
                  <img
                    src={`/api/cameras/${c.id}/stream`}
                    alt={c.name}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            ))}
            {cameras.length === 0 && (
              <div className="card text-app-muted col-span-2 text-center py-12">
                No cameras configured. Add one from the Cameras page.
              </div>
            )}
          </div>
        </div>

        {/* Right: Live Event Detections Feed (Side-by-side Vehicle + Plate) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-app-primary uppercase tracking-wider flex items-center gap-2">
              <span>⚡</span> Real-time Detections
            </h2>
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Live Stream
            </span>
          </div>

          <div className="card p-4 space-y-3 max-h-[calc(100vh-180px)] overflow-y-auto">
            {liveEvents.map((ev) => (
              <div
                key={ev.id}
                className="bg-surface-alt p-3 rounded-xl border border-app hover:border-blue-500/40 transition shadow-sm"
              >
                {/* Event Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-sm text-app-primary tracking-wide">
                      {ev.plate_number || (
                        <span className="text-app-muted text-xs">
                          {ev.result_class === 'C' ? 'No Plate Visible' : 'Not Read'}
                        </span>
                      )}
                    </span>
                    <StatusBadge status={ev.status} />
                  </div>
                  <span className="text-[11px] text-app-muted">
                    {new Date(ev.detected_at).toLocaleTimeString()}
                  </span>
                </div>

                {/* Side-by-Side Images: Vehicle Overview + Plate Crop */}
                <div className="grid grid-cols-2 gap-2 mb-2">
                  {/* Vehicle Image */}
                  <div className="flex flex-col items-center">
                    <div
                      onClick={() => setPreview(ev)}
                      className="w-full h-20 rounded-lg overflow-hidden bg-black/60 border border-app flex items-center justify-center cursor-pointer hover:opacity-90 hover:ring-2 hover:ring-blue-500 transition"
                      title="Click to zoom vehicle"
                    >
                      {ev.vehicle_image_path ? (
                        <img
                          src={`/api/media/${ev.vehicle_image_path}`}
                          alt="Vehicle"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-[10px] text-app-muted">No Vehicle</span>
                      )}
                    </div>
                    <span className="text-[9px] uppercase tracking-wider text-app-muted mt-1 font-semibold">
                      🚗 Vehicle
                    </span>
                  </div>

                  {/* Number Plate Crop */}
                  <div className="flex flex-col items-center">
                    <div
                      onClick={() => setPreview(ev)}
                      className="w-full h-20 rounded-lg overflow-hidden bg-black/60 border border-app flex items-center justify-center cursor-pointer hover:opacity-90 hover:ring-2 hover:ring-blue-500 transition"
                      title="Click to zoom plate"
                    >
                      {ev.plate_image_path ? (
                        <img
                          src={`/api/media/${ev.plate_image_path}`}
                          alt="Plate"
                          className="w-full h-full object-contain p-1"
                        />
                      ) : ev.image_path ? (
                        <img
                          src={`/api/media/${ev.image_path}`}
                          alt="Plate"
                          className="w-full h-full object-contain p-1"
                        />
                      ) : (
                        <span className="text-[10px] text-app-muted">No Plate</span>
                      )}
                    </div>
                    <span className="text-[9px] uppercase tracking-wider text-blue-400 mt-1 font-semibold">
                      🔍 Number Plate
                    </span>
                  </div>
                </div>

                {/* Event Meta Footer */}
                <div className="flex items-center justify-between text-[11px] text-app-muted pt-1 border-t border-app/50">
                  <span className="truncate max-w-[120px]">
                    {ev.camera_name || 'Gate Camera'}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="uppercase text-[10px] font-semibold text-app-secondary">
                      {ev.direction}
                    </span>
                    <span>•</span>
                    <span>{Math.round((ev.confidence || 0) * 100)}%</span>
                  </div>
                </div>
              </div>
            ))}

            {liveEvents.length === 0 && (
              <div className="text-center py-12 text-app-muted text-xs">
                Waiting for incoming vehicle passes…
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Lightbox for zooming both images */}
      {preview && (
        <ImageLightbox
          vehicleSrc={preview.vehicle_image_path ? `/api/media/${preview.vehicle_image_path}` : null}
          plateSrc={
            preview.plate_image_path
              ? `/api/media/${preview.plate_image_path}`
              : (preview.image_path ? `/api/media/${preview.image_path}` : null)
          }
          src={preview.image_path ? `/api/media/${preview.image_path}` : null}
          alt={preview.plate_number || 'no plate'}
          filename={`${preview.plate_number || 'noplate'}_${preview.id}.jpg`}
          event={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </Layout>
  )
}
