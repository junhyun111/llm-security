import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function ProtectedRoute() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="screen-center">
        <div className="loader" />
      </div>
    )
  }

  return user ? <Outlet /> : <Navigate to="/login" replace />
}
