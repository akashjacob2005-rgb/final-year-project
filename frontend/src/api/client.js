// Thin fetch wrapper. Auth travels in httpOnly cookies, so every request needs
// credentials: 'include' and there is no token for JavaScript to mishandle.

const BASE = '/api'

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message)
    this.status = status
    this.body = body
  }
}

function extractMessage(body, fallback) {
  if (!body) return fallback
  const d = body.detail
  if (typeof d === 'string') return d
  // FastAPI validation errors arrive as a list of {loc, msg, type}.
  if (Array.isArray(d) && d.length) {
    const first = d[0]
    const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : ''
    return field ? `${field}: ${first.msg}` : first.msg
  }
  return body.message || fallback
}

let refreshing = null

async function raw(path, { method = 'GET', body, isForm = false } = {}) {
  const options = {
    method,
    credentials: 'include',
    headers: isForm ? {} : { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) options.body = isForm ? body : JSON.stringify(body)

  const res = await fetch(BASE + path, options)
  if (res.status === 204) return null

  const text = await res.text()
  let parsed = null
  try {
    parsed = text ? JSON.parse(text) : null
  } catch {
    parsed = null
  }

  if (!res.ok) {
    throw new ApiError(
      extractMessage(parsed, res.statusText || 'Request failed'),
      res.status,
      parsed
    )
  }
  return parsed
}

// On a 401, try one silent refresh and replay the request. Concurrent 401s
// share a single refresh so a page with several requests in flight does not
// fire off a burst of them.
async function request(path, options = {}) {
  try {
    return await raw(path, options)
  } catch (err) {
    const isAuthCall = path.startsWith('/auth/')
    if (!(err instanceof ApiError) || err.status !== 401 || isAuthCall) throw err

    if (!refreshing) {
      refreshing = raw('/auth/refresh', { method: 'POST' }).finally(() => {
        refreshing = null
      })
    }
    try {
      await refreshing
    } catch {
      throw err
    }
    return raw(path, options)
  }
}

export const api = {
  // auth
  signup: (data) => request('/auth/signup', { method: 'POST', body: data }),
  login: (data) => request('/auth/login', { method: 'POST', body: data }),
  logout: () => request('/auth/logout', { method: 'POST' }),
  me: () => request('/auth/me'),
  updateMe: (data) => request('/auth/me', { method: 'PATCH', body: data }),

  // assessment
  startAssessment: () => request('/assessment/start', { method: 'POST' }),
  submitTest: (id, payload) =>
    request(`/assessment/${id}/test`, { method: 'POST', body: payload }),
  submitAudio: (id, formData) =>
    request(`/assessment/${id}/audio`, { method: 'POST', body: formData, isForm: true }),
  submitTranscript: (id, payload) =>
    request(`/assessment/${id}/transcript`, { method: 'POST', body: payload }),
  completeAssessment: (id) =>
    request(`/assessment/${id}/complete`, { method: 'POST' }),
  getResult: (id) => request(`/assessment/${id}`),

  // history
  sessions: () => request('/history/sessions'),
  trend: () => request('/history/trend'),
  summary: () => request('/history/summary'),
  deleteSession: (id) => request(`/history/sessions/${id}`, { method: 'DELETE' }),
  modelInfo: () => request('/history/model-info'),
}
