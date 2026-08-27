export default function ImageLightbox({ src, alt, filename, onClose }) {
  if (!src) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
      onClick={onClose}
    >
      <div
        className="relative max-w-4xl max-h-[85vh] bg-app border border-app rounded-xl p-4 flex flex-col gap-3"
        onClick={(e) => e.stopPropagation()}
      >
        <img src={src} alt={alt || ''} className="max-w-full max-h-[70vh] object-contain rounded-lg" />
        <div className="flex justify-end gap-2">
          <a
            href={src}
            download={filename || 'anpr-event.jpg'}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-app-primary text-sm font-medium"
          >
            Download
          </a>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg surface-2 hover-surface-3 text-app-primary text-sm font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
