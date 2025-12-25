// app/page.tsx  (Next.js 13+ App Router) or pages/index.tsx (Pages Router)
// Drop-in shiny frontend for our InfiniteTalk scaffold.
// - Streams WS events from BACKEND_URL/session
// - Attaches HLS video when first segment arrives (uses hls.js for Chrome)
// - Mock mode plays a demo clip if backend isn’t ready yet
// - Simple chat input that would feed GPT (stub for now)
// Styling: Tailwind; motion polish via framer-motion (optional)

'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function Page() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [wsStatus, setWsStatus] = useState<'idle' | 'connecting' | 'open' | 'closed' | 'error'>('idle')
  const [log, setLog] = useState<string[]>([])
  const [attached, setAttached] = useState(false)
  const [mockMode, setMockMode] = useState(false)
  const [idleRunning, setIdleRunning] = useState(false)
  const [prompt, setPrompt] = useState('Tell me about yourself in 1 sentence.')
  const [sending, setSending] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const hlsRef = useRef<any | null>(null)
  const isAttachingRef = useRef(false)
  const attachedRef = useRef(false)

  const appendLog = (s: string) => setLog((prev) => [s, ...prev].slice(0, 100))

  // Attach HLS source to <video> using hls.js at runtime for Chrome
  const attachHls = async () => {
    if (attached || isAttachingRef.current) return
    isAttachingRef.current = true

    // Wait a tick to ensure video element is fully mounted
    await new Promise(r => setTimeout(r, 100))

    if (!videoRef.current) {
      appendLog('Video element not found')
      isAttachingRef.current = false
      return
    }

    try {
      // @ts-ignore
      const Hls = (await import('hls.js/dist/hls.min.js')).default || (await import('hls.js')).default
      if (Hls.isSupported()) {
        if (hlsRef.current) {
          hlsRef.current.destroy()
        }

        const hls = new Hls({ maxBufferLength: 10, debug: true })
        hlsRef.current = hls

        // Load the main managed manifest (manifest.m3u8) which handles both idle and live segments
        hls.loadSource(`${BACKEND_URL}/hls/manifest.m3u8`)

        // CRITICAL: attachMedia must be called AFTER loadSource and BEFORE play
        hls.attachMedia(videoRef.current)
        appendLog('HLS attached to video element')

        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          appendLog('Manifest parsed, attempting autoplay')
          videoRef.current!.play().catch((err) => {
            appendLog('Autoplay blocked: ' + String(err))
          })
        })

        // log HLS errors/events to the UI log to aid debugging
        hls.on(Hls.Events.ERROR, (_evt: any, data: any) => {
          const msg = 'HLS ERROR: ' + (data?.type || '') + ' ' + JSON.stringify(data)
          appendLog(msg)
          try { if (wsRef.current && wsRef.current.readyState === 1) wsRef.current.send(JSON.stringify({ type: 'client_log', level: 'error', msg })) } catch (e) { }
        })
        hls.on(Hls.Events.LOG, (_evt: any, data: any) => {
          const msg = 'HLS LOG: ' + JSON.stringify(data)
          appendLog(msg)
          try { if (wsRef.current && wsRef.current.readyState === 1) wsRef.current.send(JSON.stringify({ type: 'client_log', level: 'log', msg })) } catch (e) { }
        })

        setAttached(true)
        attachedRef.current = true
      } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
        videoRef.current.src = `${BACKEND_URL}/hls/manifest.m3u8`
        setAttached(true)
        appendLog('Using native HLS playback (Safari)')
      } else {
        appendLog('HLS not supported. Consider a Chrome extension or Safari.')
      }
    } catch (e) {
      appendLog('Failed to load hls.js: ' + String(e))
    } finally {
      isAttachingRef.current = false
    }
  }

  // WebSocket lifecycle
  useEffect(() => {
    if (mockMode) return // don’t connect in mock mode
    setWsStatus('connecting')
    const ws = new WebSocket(BACKEND_URL.replace('http', 'ws') + '/session')
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('open')
      appendLog('WS open')
    }
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'video_segment') {
          const idx = msg.index !== undefined ? msg.index : '?'
          appendLog(`Segment #${idx} ready: ${msg.uri}`)
          if (!attachedRef.current) {
            attachHls()
            // attachedRef.current will be set true inside attachHls
          }
        } else if (msg.type === 'status') {
          appendLog(`Status: ${msg.stage}`)
          // reflect idle state reported from server
          if (msg.stage === 'idle_running' || msg.stage === 'idle_setup') setIdleRunning(true)
          if (msg.stage === 'idle_stopped' || msg.stage === 'idle_setup_failed' || msg.stage === 'idle_missing') setIdleRunning(false)
        } else if (msg.type === 'error') {
          appendLog(`ERROR: ${msg.message}`)
        } else if (msg.type === 'partial_text') {
          appendLog(`LLM: ${msg.text}`)
        }
      } catch {
        appendLog('WS message: ' + ev.data)
      }
    }
    ws.onerror = () => {
      setWsStatus('error')
      appendLog('WS error')
    }
    ws.onclose = () => {
      setWsStatus('closed')
      appendLog('WS closed')
    }
    return () => {
      ws.close()
      attachedRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mockMode])

  const toggleIdle = () => {
    if (!wsRef.current || wsRef.current.readyState !== 1) {
      appendLog('WS not open; cannot toggle idle')
      return
    }
    if (idleRunning) {
      wsRef.current.send(JSON.stringify({ type: 'control', action: 'stop_idle' }))
      appendLog('Sent stop_idle')
    } else {
      wsRef.current.send(JSON.stringify({ type: 'control', action: 'start_idle' }))
      appendLog('Sent start_idle')
    }
  }

  // Mock mode video source
  useEffect(() => {
    if (!videoRef.current) return
    if (mockMode) {
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }
      setAttached(true)
      videoRef.current.src = '/demo/out.mp4' // place a demo clip under public/demo/out.mp4
      videoRef.current.play().catch(() => { })
    } else {
      setAttached(false)
      attachedRef.current = false
      videoRef.current.src = ''
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }
    }
  }, [mockMode])

  const onSend = async () => {
    if (!wsRef.current || wsRef.current.readyState !== 1) {
      appendLog('WS not open; cannot send prompt')
      return
    }
    setSending(true)
    try {
      wsRef.current.send(JSON.stringify({ type: 'prompt', text: prompt }))
      appendLog('Prompt sent to backend')
    } catch (e) {
      appendLog('Send failed: ' + String(e))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col items-center p-6">
      <div className="w-full max-w-5xl grid gap-6">
        <header className="flex items-center justify-between">
          <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="text-2xl md:text-3xl font-semibold">
            <span className="text-emerald-400">Digital Creations Windsor</span> Colin's Live Avatar(20years ago)
          </motion.h1>
          <div className="flex items-center gap-3 text-sm">
            <span className={`rounded-full w-2.5 h-2.5 ${wsStatus === 'open' ? 'bg-emerald-400' : wsStatus === 'connecting' ? 'bg-yellow-400' : 'bg-zinc-600'}`}></span>
            <span className="hidden sm:inline">WS: {wsStatus}</span>
            <button onClick={toggleIdle} className={`ml-3 px-2 py-1 rounded ${idleRunning ? 'bg-red-600' : 'bg-emerald-600'}`}>{idleRunning ? 'Stop Idle' : 'Start Idle'}</button>
            <label className="flex items-center gap-2 ml-4">
              <input type="checkbox" checked={mockMode} onChange={(e) => setMockMode(e.target.checked)} />
              Video mode
            </label>
          </div>
        </header>

        <section className="grid md:grid-cols-2 gap-6">
          <div className="bg-zinc-900/60 rounded-2xl p-4 border border-zinc-800">
            <div className="aspect-video overflow-hidden rounded-xl bg-zinc-950/60 flex items-center justify-center">
              <video ref={videoRef} autoPlay loop controls playsInline muted className="w-full h-full" />
            </div>
            <p className="mt-3 text-xs text-zinc-400">Video attaches when first segment is ready. In Chrome we load hls.js automatically.</p>
          </div>

          <div className="bg-zinc-900/60 rounded-2xl p-4 border border-zinc-800 flex flex-col gap-3">
            <div>
              <label className="text-sm text-zinc-400">Prompt</label>
              <div className="mt-1 flex gap-2">
                <input value={prompt} onChange={(e) => setPrompt(e.target.value)} className="w-full px-3 py-2 rounded-xl bg-zinc-800 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-emerald-500" />
                <button onClick={onSend} disabled={sending} className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-medium disabled:opacity-50">Send</button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm text-zinc-400">Reference Image</label>
                <input type="file" accept="image/*" className="mt-1 block w-full text-sm" />
              </div>
              <div>
                <label className="text-sm text-zinc-400">Audio</label>
                <input type="file" accept="audio/*" className="mt-1 block w-full text-sm" />
              </div>
            </div>

            <div className="mt-2 text-xs text-zinc-400">Uploaders are UI only in this stub; backend will pick up mic/chat → TTS stream.
            </div>

            <div className="mt-4 h-40 overflow-auto bg-black/30 rounded-xl p-3 text-xs border border-zinc-800">
              {log.length === 0 ? <div className="text-zinc-500">Waiting for events…</div> : log.map((l, i) => (
                <div key={i} className="text-zinc-300">{l}</div>
              ))}
            </div>
          </div>
        </section>

        <footer className="text-center text-xs text-zinc-500">
          Backend: <span className="text-zinc-300">{BACKEND_URL}</span> · Toggle <span className="text-zinc-300">Mock mode</span> to demo without backend.
        </footer>
      </div>
    </div>
  )
}


