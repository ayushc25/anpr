import StatusBadge from './StatusBadge'

export default function ImageLightbox({
  src,
  vehicleSrc,
  plateSrc,
  alt,
  filename,
  event,
  onClose,
}) {
  const vSrc = vehicleSrc || (src && !plateSrc ? src : null)
  const pSrc = plateSrc || (src && !vehicleSrc ? src : null)

  if (!vSrc && !pSrc && !src) return null

  const hasBoth = Boolean(vSrc && pSrc && vSrc !== pSrc)

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 sm:p-6 backdrop-blur-xs"
      onClick={onClose}
    >
      <div
        className={`relative w-full ${
          hasBoth ? 'max-w-5xl' : 'max-w-3xl'
        } max-h-[90vh] bg-app border border-app rounded-2xl p-5 flex flex-col gap-4 shadow-2xl overflow-y-auto`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-app">
          <div className="flex items-center gap-3">
            <span className="text-xl">📷</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold text-lg text-app-primary">
                  {event?.plate_number || alt || 'Vehicle Event'}
                </span>
                {event?.status && <StatusBadge status={event.status} />}
              </div>
              <div className="text-xs text-app-muted flex items-center gap-2 mt-0.5">
                {event?.camera_name && <span>Camera: {event.camera_name}</span>}
                {event?.camera_name && event?.detected_at && <span>•</span>}
                {event?.detected_at && <span>{new Date(event.detected_at).toLocaleString()}</span>}
                {event?.confidence != null && (
                  <>
                    <span>•</span>
                    <span>Confidence: {Math.round(event.confidence * 100)}%</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg surface-2 hover-surface-3 flex items-center justify-center text-app-secondary hover:text-app-primary transition"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Image Display: Side-by-Side if both available */}
        <div className={`grid ${hasBoth ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'} gap-4`}>
          {/* Vehicle Image */}
          {vSrc && (
            <div className="flex flex-col gap-2 bg-surface-alt p-3 rounded-xl border border-app">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-app-secondary uppercase tracking-wider flex items-center gap-1.5">
                  🚗 Vehicle Overview
                </span>
                <a
                  href={vSrc}
                  download={`vehicle_${filename || 'event.jpg'}`}
                  className="text-xs text-blue-400 hover:text-blue-300 font-medium"
                >
                  Download
                </a>
              </div>
              <div className="rounded-lg overflow-hidden bg-black/60 flex items-center justify-center min-h-[220px] max-h-[50vh]">
                <img
                  src={vSrc}
                  alt="Vehicle Overview"
                  className="max-w-full max-h-[50vh] object-contain rounded"
                />
              </div>
            </div>
          )}

          {/* Number Plate Crop */}
          {pSrc && (
            <div className="flex flex-col gap-2 bg-surface-alt p-3 rounded-xl border border-app">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider flex items-center gap-1.5">
                  🔍 Cropped Number Plate
                </span>
                <a
                  href={pSrc}
                  download={`plate_${filename || 'crop.jpg'}`}
                  className="text-xs text-blue-400 hover:text-blue-300 font-medium"
                >
                  Download
                </a>
              </div>
              <div className="rounded-lg overflow-hidden bg-black/60 flex items-center justify-center min-h-[220px] max-h-[50vh]">
                <img
                  src={pSrc}
                  alt="Cropped Number Plate"
                  className="max-w-full max-h-[50vh] object-contain rounded scale-105"
                />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end pt-2 border-t border-app">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-surface hover:bg-hover-2 text-app-primary text-sm font-medium border border-app transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
