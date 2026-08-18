import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  // `loading` guards the first render only. Without it, ProtectedRoute would
  // bounce an already-signed-in user to /login before /auth/me resolves.
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    api
      .me()
      .then((u) => !cancelled && setUser(u))
      .catch(() => !cancelled && setUser(null))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (credentials) => {
    const { user: u } = await api.login(credentials)
    setUser(u)
    return u
  }, [])

  const signup = useCallback(async (data) => {
    const { user: u } = await api.signup(data)
    setUser(u)
    return u
  }, [])

  const logout = useCallback(async () => {
    try {
      await api.logout()
    } finally {
      setUser(null)
    }
  }, [])

  const updateProfile = useCallback(async (data) => {
    const u = await api.updateMe(data)
    setUser(u)
    return u
  }, [])

  return (
    <AuthContext.Provider
      value={{ user, loading, login, signup, logout, updateProfile }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
