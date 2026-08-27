import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const ITEMS = [
  { to: '/', label: 'Dashboard', icon: '🏠', permission: 'dashboard' },
  { to: '/live', label: 'Live View', icon: '🎥', permission: 'live' },
  { to: '/events', label: 'ANPR Events', icon: '📋', permission: 'events' },
  { to: '/vehicles', label: 'Vehicle Search', icon: '🔍', permission: 'vehicles' },
  { to: '/lists', label: 'Blacklist / Whitelist', icon: '🛡️', permission: 'lists' },
  { to: '/cameras', label: 'Cameras', icon: '📷', permission: 'cameras' },
  { to: '/locations', label: 'Locations', icon: '📍', permission: 'locations' },
  { to: '/reports', label: 'Reports', icon: '📊', permission: 'reports' },
  { to: '/users', label: 'Users', icon: '👥', permission: 'users' },
  { to: '/logs', label: 'Logs', icon: '🕓', permission: 'logs' },
]

export default function Sidebar() {
  const { user } = useAuth()
  const permissions = user?.permissions || []

  return (
    <aside className="w-60 shrink-0 bg-app-alt border-r border-app-soft flex flex-col h-screen sticky top-0">
      <div className="flex items-center gap-2 px-5 py-5 border-b border-app-soft">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-lg">📷</div>
        <div>
          <div className="font-bold text-app-primary leading-tight">ANPR</div>
          <div className="text-[10px] text-app-muted tracking-widest">SYSTEM</div>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto py-3">
        {ITEMS.filter((i) => permissions.includes(i.permission)).map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-5 py-2.5 text-sm mx-2 rounded-lg mb-0.5 transition ${
                isActive ? 'bg-blue-600/15 text-blue-400' : 'text-app-secondary hover-surface-1 hover-text-primary'
              }`
            }
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-app-soft flex items-center gap-3">
        <div className="w-9 h-9 rounded-full bg-gray-700 flex items-center justify-center">👤</div>
        <div className="text-sm">
          <div className="text-app-primary font-medium">{user?.full_name || user?.username}</div>
          <div className="text-app-muted text-xs">{user?.role_name || ''}</div>
        </div>
      </div>
    </aside>
  )
}
