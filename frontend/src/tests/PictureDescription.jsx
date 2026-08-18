import { useEffect, useRef, useState } from 'react'

/**
 * Test 2 — Picture description (the machine-learned component).
 *
 * The user describes a kitchen scene aloud. The recording is transcribed by
 * Whisper on the server and scored by the classifier trained on DementiaBank
 * Cookie Theft transcripts.
 *
 * A typed fallback exists because getUserMedia needs HTTPS (or localhost) and a
 * granted microphone permission, and neither is guaranteed on a demo machine.
 * Losing the whole assessment to a blocked mic prompt is not acceptable.
 */
export default function PictureDescription({ spec, sessionId, api, onComplete, onError }) {
  const minSeconds = spec?.min_seconds ?? 20
  const maxSeconds = spec?.max_seconds ?? 120

  const [mode, setMode] = useState('idle') // idle | recording | uploading | typing
  const [elapsed, setElapsed] = useState(0)
  const [text, setText] = useState('')
  const [notice, setNotice] = useState('')
  const [supported, setSupported] = useState(true)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const startRef = useRef(null)

  useEffect(() => {
    const ok =
      typeof navigator !== 'undefined' &&
      navigator.mediaDevices?.getUserMedia &&
      typeof window.MediaRecorder !== 'undefined'
    setSupported(Boolean(ok))
    if (!ok) setMode('typing')
  }, [])

  // Recording timer, with a hard stop at the maximum duration.
  useEffect(() => {
    if (mode !== 'recording') return
    const t = setInterval(() => {
      const secs = Math.floor((Date.now() - startRef.current) / 1000)
      setElapsed(secs)
      if (secs >= maxSeconds) stopRecording()
    }, 250)
    return () => clearInterval(t)
  }, [mode, maxSeconds])

  // Never leave the microphone open if the component goes away mid-recording.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  async function startRecording() {
    setNotice('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []

      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
        .find((m) => MediaRecorder.isTypeSupported(m)) || ''

      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
      rec.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data)
      rec.onstop = handleStop
      recorderRef.current = rec

      startRef.current = Date.now()
      setElapsed(0)
      rec.start()
      setMode('recording')
    } catch {
      setSupported(false)
      setMode('typing')
      setNotice(
        'The microphone is unavailable, so you can type your description instead.'
      )
    }
  }

  function stopRecording() {
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') rec.stop()
  }

  async function handleStop() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    const durationMs = Date.now() - startRef.current
    const blob = new Blob(chunksRef.current, {
      type: recorderRef.current?.mimeType || 'audio/webm',
    })

    if (!blob.size) {
      setMode('idle')
      setNotice('Nothing was recorded. Please try again.')
      return
    }

    setMode('uploading')
    const form = new FormData()
    form.append('file', blob, 'description.webm')
    form.append('duration_ms', String(durationMs))

    try {
      const res = await api.submitAudio(sessionId, form)
      if (res?.language && res.language.available === false) {
        setMode('idle')
        setNotice(
          `${res.language.reason || 'Not enough speech was captured.'} Please record again, or type your description.`
        )
        return
      }
      onComplete({ recorded: true, transcript: res?.transcript })
    } catch (err) {
      setMode('idle')
      setNotice(err.message || 'The recording could not be uploaded.')
      onError?.(err)
    }
  }

  async function submitTyped() {
    setMode('uploading')
    try {
      const res = await api.submitTranscript(sessionId, {
        transcript: text,
        duration_ms: null,
      })
      if (res?.language && res.language.available === false) {
        setMode('typing')
        setNotice(res.language.reason || 'Please write a little more.')
        return
      }
      onComplete({ recorded: false })
    } catch (err) {
      setMode('typing')
      setNotice(err.message || 'Could not submit your description.')
      onError?.(err)
    }
  }

  const canStop = elapsed >= minSeconds
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  return (
    <div className="card test-card">
      <div className="test-head">
        <div className="test-name">2 · Picture Description</div>
        <div className="test-instruction">
          {spec?.prompt || 'Tell me everything you see going on in this picture.'}
        </div>
      </div>

      <div className="test-body">
        <div className="scene-frame">
          <img src={spec?.image || '/assets/kitchen-scene.svg'} alt="Kitchen scene to describe" />
        </div>

        {notice && <div className="alert alert-warn">{notice}</div>}

        {mode === 'idle' && (
          <div className="text-center">
            <p className="muted mb-2">
              Speak for at least {minSeconds} seconds. Describe the people, what
              they are doing, and anything else you notice.
            </p>
            <button className="btn btn-primary btn-lg" onClick={startRecording}>
              ● Start recording
            </button>
            <div className="mt-2">
              <button className="btn btn-ghost small" onClick={() => setMode('typing')}>
                Type it instead
              </button>
            </div>
          </div>
        )}

        {mode === 'recording' && (
          <div className="record-row">
            <span className="rec-dot" />
            <span className="rec-timer">
              {String(Math.floor(elapsed / 60)).padStart(2, '0')}:
              {String(elapsed % 60).padStart(2, '0')}
            </span>
            <button
              className="btn btn-primary"
              onClick={stopRecording}
              disabled={!canStop}
            >
              {canStop
                ? 'Stop and submit'
                : `Keep going (${minSeconds - elapsed}s)`}
            </button>
          </div>
        )}

        {mode === 'uploading' && (
          <div className="text-center">
            <div className="spinner mb-2" />
            <p className="muted small">
              Transcribing and analysing your description…
            </p>
          </div>
        )}

        {mode === 'typing' && (
          <div>
            <textarea
              className="input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Describe everything you see happening in the picture…"
            />
            <div className="row between mt-1">
              <span className="hint">
                {wordCount} words · at least {spec?.min_words ?? 25} needed
              </span>
              <div className="row">
                {supported && (
                  <button className="btn btn-ghost" onClick={() => { setMode('idle'); setNotice('') }}>
                    Record instead
                  </button>
                )}
                <button
                  className="btn btn-primary"
                  onClick={submitTyped}
                  disabled={wordCount < (spec?.min_words ?? 25)}
                >
                  Submit
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="test-foot">
        <span className="muted small">
          Your description is analysed by a model trained on clinical transcripts
        </span>
      </div>
    </div>
  )
}
