import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  // Wait for the initial /auth/me to settle, otherwise a signed-in user gets
  // redirected to /login on every hard refresh.
  if (loading) {
    return (
      <div className="center-screen">
        <div>
          <div className="spinner" />
          <p className="muted small mt-2">Loading…</p>
        </div>
      </div>
    )
  }

  if (!user) {
    // Remember where they were headed so login can send them back.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return children
}
