import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from 'react'
import { api, ApiError } from '../api'
import type { User } from '../types'

type AuthContextValue = {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .get<User>('/api/auth/me')
      .then(setUser)
      .catch((error) => {
        if (!(error instanceof ApiError && error.status === 401)) {
          console.error(error)
        }
        setUser(null)
      })
      .finally(() => setLoading(false))
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      login: async (email, password) => {
        const result = await api.post<User>('/api/auth/login', { email, password })
        setUser(result)
      },
      register: async (email, password, displayName) => {
        const result = await api.post<User>('/api/auth/register', {
          email,
          password,
          displayName
        })
        setUser(result)
      },
      logout: async () => {
        await api.post<void>('/api/auth/logout')
        setUser(null)
      }
    }),
    [user, loading]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
