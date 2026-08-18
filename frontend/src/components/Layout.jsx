import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

const NAV = [
  { to: '/dashboard', label: 'Dashboard', icon: '◈' },
  { to: '/assessment', label: 'New Assessment', icon: '⊕' },
  { to: '/history', label: 'Monitoring', icon: '◔' },
  { to: '/model', label: 'Model & Method', icon: '⚙' },
  { to: '/profile', label: 'Profile', icon: '◯' },
]

const TITLES = {
  '/dashboard': ['Dashboard', 'Your cognitive health at a glance'],
  '/assessment': ['Cognitive Assessment', 'Six tests, about ten minutes'],
  '/history': ['Longitudinal Monitoring', 'How your scores change over time'],
  '/model': ['Model & Method', 'How the score is produced, and how well it performs'],
  '/profile': ['Profile', 'Details used to age- and education-adjust your scores'],
}

function initials(name = '') {
  return name.trim().split(/\s+/).slice(0, 2).map((p) => p[0]?.toUpperCase()).join('') || '?'
}

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [open, setOpen] = useState(false)

  const [title, subtitle] = TITLES[location.pathname] ||
    (location.pathname.startsWith('/results')
      ? ['Assessment Result', 'Your screening outcome and what drove it']
      : ['NeuroGuard', ''])

  async function handleLogout() {
    await logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand">
          <div className="brand-mark">NG</div>
          <div>
            <div className="brand-name">NeuroGuard</div>
            <div className="brand-sub">Cognitive Screening</div>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={() => setOpen(false)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="user-chip">
            <div className="avatar">{initials(user?.full_name)}</div>
            <div style={{ minWidth: 0 }}>
              <div className="user-name">{user?.full_name}</div>
              <div className="user-meta">Age {user?.age}</div>
            </div>
          </div>
          <button className="btn btn-ghost btn-block" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div>
            <h1>{title}</h1>
            {subtitle && <div className="topbar-sub">{subtitle}</div>}
          </div>
          <button
            className="btn btn-secondary"
            style={{ display: 'none' }}
            onClick={() => setOpen((v) => !v)}
          >
            Menu
          </button>
        </header>
        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
