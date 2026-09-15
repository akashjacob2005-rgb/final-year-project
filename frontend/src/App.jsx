import { useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import { initTheme } from './theme.js'
import ProtectedRoute from './components/ProtectedRoute.jsx'
import Assessment from './pages/Assessment.jsx'
import Dashboard from './pages/Dashboard.jsx'
import ForgotPassword from './pages/ForgotPassword.jsx'
import History from './pages/History.jsx'
import Login from './pages/Login.jsx'
import ModelInfo from './pages/ModelInfo.jsx'
import MriAnalysis from './pages/MriAnalysis.jsx'
import Profile from './pages/Profile.jsx'
import ResetPassword from './pages/ResetPassword.jsx'
import Results from './pages/Results.jsx'
import Signup from './pages/Signup.jsx'

export default function App() {
  useEffect(initTheme, [])

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      {/* Public: reached while signed out, by definition. */}
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/reset-password" element={<ResetPassword />} />

      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/assessment" element={<Assessment />} />
        <Route path="/results/:sessionId" element={<Results />} />
        <Route path="/history" element={<History />} />
        <Route path="/model" element={<ModelInfo />} />
        <Route path="/mri" element={<MriAnalysis />} />
        <Route path="/profile" element={<Profile />} />
      </Route>

      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  )
}
