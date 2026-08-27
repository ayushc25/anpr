const COLORS = {
  blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  green: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  red: 'text-red-400 bg-red-500/10 border-red-500/20',
  yellow: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
}

export default function StatCard({ icon, label, value, sub, color = 'blue' }) {
  return (
    <div className={`card p-4 border ${COLORS[color]}`}>
      <div className="flex items-center gap-2 text-sm text-app-secondary mb-2">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <div className="text-2xl font-bold text-app-primary">{value}</div>
      {sub && <div className="text-xs text-app-muted mt-1">{sub}</div>}
    </div>
  )
}
