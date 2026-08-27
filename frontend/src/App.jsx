import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import LiveView from './pages/LiveView'
import Events from './pages/Events'
import VehicleSearch from './pages/VehicleSearch'
import Lists from './pages/Lists'
import Cameras from './pages/Cameras'
import Locations from './pages/Locations'
import Reports from './pages/Reports'
import Users from './pages/Users'
import Logs from './pages/Logs'

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={<ProtectedRoute permission="dashboard"><Dashboard /></ProtectedRoute>} />
            <Route path="/live" element={<ProtectedRoute permission="live"><LiveView /></ProtectedRoute>} />
            <Route path="/events" element={<ProtectedRoute permission="events"><Events /></ProtectedRoute>} />
            <Route path="/vehicles" element={<ProtectedRoute permission="vehicles"><VehicleSearch /></ProtectedRoute>} />
            <Route path="/lists" element={<ProtectedRoute permission="lists"><Lists /></ProtectedRoute>} />
            <Route path="/cameras" element={<ProtectedRoute permission="cameras"><Cameras /></ProtectedRoute>} />
            <Route path="/locations" element={<ProtectedRoute permission="locations"><Locations /></ProtectedRoute>} />
            <Route path="/reports" element={<ProtectedRoute permission="reports"><Reports /></ProtectedRoute>} />
            <Route path="/users" element={<ProtectedRoute permission="users"><Users /></ProtectedRoute>} />
            <Route path="/logs" element={<ProtectedRoute permission="logs"><Logs /></ProtectedRoute>} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  )
}
