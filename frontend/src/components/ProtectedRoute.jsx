import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function ProtectedRoute({ children, permission }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  if (permission && !(user.permissions || []).includes(permission)) {
    return (
      <div className="flex items-center justify-center h-screen text-app-secondary text-sm">
        You don't have access to this page.
      </div>
    )
  }
  return children
}
