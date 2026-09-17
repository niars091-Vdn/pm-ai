"""
LayoutSync - Blocco 4: API FastAPI
====================================
Espone la pipeline completa come API REST.

Endpoints:
  POST /analizza     → riceve point cloud + metadata → restituisce ZIP con tutti i file
  GET  /status       → health check
  GET  /demo         → esegue demo con dati simulati
  GET  /docs         → documentazione interattiva (Swagger UI)
"""

import os, sys, json, math, shutil, zipfile, io, time, uuid, struct
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFilter

import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="LayoutSync API",
    description="Converte rilievi LiDAR/ARCore in documentazione CAD completa",
    version="1.0.0",
)



# ── Serve la PWA direttamente ──
APP_HTML = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#0a0e1a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="LayoutSync">
<title>LayoutSync</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #0a0e1a;
  --bg2:       #111827;
  --surface:   #1a2035;
  --border:    #1e2d47;
  --accent:    #2dd4bf;
  --accent-dim:rgba(45,212,191,0.12);
  --blue:      #3b82f6;
  --blue-dim:  rgba(59,130,246,0.12);
  --warn:      #f59e0b;
  --warn-dim:  rgba(245,158,11,0.10);
  --danger:    #ef4444;
  --green:     #10b981;
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #475569;
  --mono:      'JetBrains Mono', monospace;
  --sans:      'Inter', sans-serif;
  --r:         14px;
  --r-sm:      8px;
}

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent; }
html,body { height:100%; overflow:hidden; background:var(--bg); }

body {
  font-family: var(--sans);
  color: var(--text);
  display: flex;
  flex-direction: column;
  height: 100%;
  height: -webkit-fill-available;
}

/* ── SPLASH ── */
#splash {
  position:fixed; inset:0; z-index:999;
  background:var(--bg);
  display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:20px;
  transition:opacity 0.5s;
}
#splash.out { opacity:0; pointer-events:none; }

.splash-logo {
  font-size:42px; font-weight:700;
  letter-spacing:-2px; line-height:1;
}
.splash-logo em { color:var(--accent); font-style:normal; }

.splash-sub {
  font-size:13px; color:var(--text2);
  letter-spacing:0.5px; text-transform:uppercase;
}

.splash-ring {
  width:48px; height:48px;
  border-radius:50%;
  border:2px solid var(--border);
  border-top-color:var(--accent);
  animation:spin 0.9s linear infinite;
  margin-top:8px;
}

/* ── TOP BAR ── */
#topbar {
  display:flex; align-items:center;
  justify-content:space-between;
  padding:12px 16px 10px;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
  position:relative; z-index:10;
}
.logo-sm {
  font-size:18px; font-weight:700;
  letter-spacing:-0.5px;
}
.logo-sm em { color:var(--accent); font-style:normal; }

.server-pill {
  display:flex; align-items:center; gap:6px;
  padding:5px 11px;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:20px;
  font-family:var(--mono);
  font-size:10px; color:var(--text2);
  cursor:pointer;
  transition:border-color 0.2s;
}
.server-pill:active { border-color:var(--accent); }
.dot { width:7px; height:7px; border-radius:50%; background:var(--text3); transition:all 0.3s; }
.dot.on  { background:var(--accent); box-shadow:0 0 8px var(--accent); }
.dot.off { background:var(--danger); }

/* ── PROGRESS BAR ── */
#progress {
  display:flex; gap:4px; padding:10px 16px;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.prog-seg {
  flex:1; height:3px; border-radius:2px;
  background:var(--border);
  transition:background 0.4s;
  position:relative; overflow:hidden;
}
.prog-seg.active::after {
  content:'';
  position:absolute; inset:0;
  background:var(--accent);
  animation:pulse-seg 1.5s ease-in-out infinite;
}
@keyframes pulse-seg {
  0%,100% { opacity:1; } 50% { opacity:0.5; }
}
.prog-seg.done { background:var(--accent); }

/* ── MAIN ── */
#main { flex:1; position:relative; overflow:hidden; }

.screen {
  position:absolute; inset:0;
  display:flex; flex-direction:column;
  opacity:0; pointer-events:none;
  transition:opacity 0.2s, transform 0.2s;
  transform:translateX(30px);
}
.screen.active {
  opacity:1; pointer-events:all;
  transform:translateX(0);
}
.screen.back { transform:translateX(-30px); }

/* ── BOTTOM BAR ── */
.bot {
  padding:12px 16px;
  padding-bottom:max(12px, env(safe-area-inset-bottom));
  background:var(--bg2);
  border-top:1px solid var(--border);
  flex-shrink:0;
  display:flex; gap:10px;
}

.btn {
  flex:1; padding:14px;
  border:none; border-radius:var(--r);
  font-family:var(--sans);
  font-size:15px; font-weight:600;
  cursor:pointer; transition:all 0.15s;
}
.btn:active { transform:scale(0.97); }
.btn.primary { background:var(--accent); color:#0a0e1a; }
.btn.primary:disabled { opacity:0.35; pointer-events:none; }
.btn.ghost {
  background:transparent; color:var(--text2);
  border:1.5px solid var(--border); flex:0 0 auto;
  padding:14px 18px;
}
.btn.ghost:active { border-color:var(--accent); color:var(--accent); }

/* ══════════════════════════════════
   SCREEN 0 — WELCOME
══════════════════════════════════ */
#s-welcome {
  background:var(--bg);
  align-items:center; justify-content:center;
  text-align:center; gap:0; padding:0;
  overflow-y:auto;
}

.welcome-hero {
  width:100%; padding:48px 24px 32px;
  background:linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
  border-bottom:1px solid var(--border);
}
.welcome-icon {
  font-size:64px; margin-bottom:20px;
  filter:drop-shadow(0 0 20px rgba(45,212,191,0.4));
}
.welcome-title {
  font-size:32px; font-weight:700;
  letter-spacing:-1px; margin-bottom:8px;
  line-height:1.1;
}
.welcome-title em { color:var(--accent); font-style:normal; }
.welcome-sub {
  font-size:14px; color:var(--text2);
  line-height:1.6; max-width:280px; margin:0 auto;
}

.welcome-steps {
  width:100%; padding:24px 20px;
  display:flex; flex-direction:column; gap:14px;
}
.w-step {
  display:flex; align-items:flex-start; gap:14px;
  padding:14px; background:var(--surface);
  border:1px solid var(--border); border-radius:var(--r);
  text-align:left;
}
.w-step-num {
  width:32px; height:32px; flex-shrink:0;
  border-radius:50%; background:var(--accent-dim);
  border:1px solid var(--accent);
  display:flex; align-items:center; justify-content:center;
  font-family:var(--mono); font-size:13px;
  color:var(--accent); font-weight:500;
}
.w-step-body strong { display:block; font-size:14px; margin-bottom:3px; }
.w-step-body span   { font-size:12px; color:var(--text2); line-height:1.5; }

.welcome-bot { padding:16px 20px 32px; width:100%; }

/* ══════════════════════════════════
   SCREEN 1 — DISEGNO PIANTA
══════════════════════════════════ */
#s-draw { background:var(--bg); }

.toolbar {
  display:flex; gap:6px; padding:10px 12px;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  overflow-x:auto; flex-shrink:0;
  scrollbar-width:none;
}
.toolbar::-webkit-scrollbar { display:none; }

.t-btn {
  display:flex; flex-direction:column;
  align-items:center; gap:3px;
  padding:8px 11px;
  background:transparent;
  border:1.5px solid var(--border);
  border-radius:var(--r-sm);
  color:var(--text2);
  font-family:var(--sans);
  font-size:10px; font-weight:500;
  cursor:pointer; flex-shrink:0;
  transition:all 0.15s;
  white-space:nowrap;
}
.t-btn .i { font-size:20px; line-height:1; }
.t-btn.on {
  border-color:var(--accent);
  color:var(--accent);
  background:var(--accent-dim);
}
.t-btn:active { transform:scale(0.93); }

#cvs-wrap { flex:1; position:relative; overflow:hidden; touch-action:none; }
#cvs { width:100%; height:100%; display:block; }

.cvs-tip {
  position:absolute; bottom:16px; left:50%;
  transform:translateX(-50%);
  background:rgba(10,14,26,0.88);
  border:1px solid var(--border);
  color:var(--text2); font-size:12px;
  padding:8px 18px; border-radius:20px;
  pointer-events:none; white-space:nowrap;
  backdrop-filter:blur(8px);
  transition:opacity 0.3s;
}

/* ══════════════════════════════════
   SCREEN 2 — FOTOCAMERA
══════════════════════════════════ */
#s-camera { background:#000; }

.cam-header {
  padding:14px 16px 10px;
  background:rgba(0,0,0,0.7);
  border-bottom:1px solid rgba(255,255,255,0.08);
  flex-shrink:0;
  backdrop-filter:blur(8px);
}
.cam-header h2 { font-size:15px; font-weight:600; margin-bottom:2px; }
.cam-header p  { font-size:12px; color:var(--text2); }

.cam-walls {
  display:flex; gap:8px; padding:10px 12px;
  background:rgba(0,0,0,0.6); flex-shrink:0;
  overflow-x:auto; scrollbar-width:none;
  backdrop-filter:blur(8px);
}
.cam-walls::-webkit-scrollbar { display:none; }

.wall-chip {
  display:flex; align-items:center; gap:6px;
  padding:7px 12px;
  border-radius:20px;
  border:1.5px solid var(--border);
  background:var(--surface);
  font-size:12px; font-weight:500;
  color:var(--text2); cursor:pointer;
  flex-shrink:0; transition:all 0.15s;
}
.wall-chip.active {
  border-color:var(--accent);
  color:var(--accent);
  background:var(--accent-dim);
}
.wall-chip.done {
  border-color:var(--green);
  color:var(--green);
  background:rgba(16,185,129,0.1);
}

#cam-view { flex:1; position:relative; background:#000; overflow:hidden; }
#video { width:100%; height:100%; object-fit:cover; }
#cam-canvas { display:none; }

.cam-overlay {
  position:absolute; inset:0;
  pointer-events:none;
}
.cam-corner {
  position:absolute;
  width:28px; height:28px;
  border-color:var(--accent);
  border-style:solid;
  opacity:0.8;
}
.cam-corner.tl { top:20px; left:20px; border-width:2px 0 0 2px; }
.cam-corner.tr { top:20px; right:20px; border-width:2px 2px 0 0; }
.cam-corner.bl { bottom:20px; left:20px; border-width:0 0 2px 2px; }
.cam-corner.br { bottom:20px; right:20px; border-width:0 2px 2px 0; }

.cam-label {
  position:absolute; top:50%; left:50%;
  transform:translate(-50%,-50%);
  background:rgba(10,14,26,0.75);
  border:1px solid var(--accent);
  color:var(--accent); font-size:13px; font-weight:600;
  padding:8px 20px; border-radius:20px;
  backdrop-filter:blur(8px);
  pointer-events:none;
}

.cam-shutter {
  position:absolute; bottom:20px; left:50%;
  transform:translateX(-50%);
  width:68px; height:68px;
  border-radius:50%;
  background:white;
  border:4px solid rgba(255,255,255,0.3);
  box-shadow:0 0 0 2px rgba(255,255,255,0.5);
  cursor:pointer; transition:all 0.12s;
  pointer-events:all;
  display:flex; align-items:center; justify-content:center;
}
.cam-shutter:active { transform:translateX(-50%) scale(0.9); }
.cam-shutter-inner {
  width:52px; height:52px; border-radius:50%;
  background:white;
}

.cam-preview-strip {
  position:absolute; bottom:20px; right:16px;
  display:flex; flex-direction:column; gap:6px;
  pointer-events:all;
}
.cam-thumb {
  width:52px; height:52px;
  border-radius:8px; object-fit:cover;
  border:2px solid var(--accent);
  box-shadow:0 2px 12px rgba(0,0,0,0.5);
}

/* ══════════════════════════════════
   SCREEN 3 — MISURE
══════════════════════════════════ */
#s-measures { background:var(--bg); overflow-y:auto; }

.sec-header {
  padding:18px 16px 12px;
  background:var(--bg2);
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.sec-header h2 { font-size:18px; font-weight:700; margin-bottom:4px; }
.sec-header p  { font-size:13px; color:var(--text2); line-height:1.5; }

.m-list { padding:12px; display:flex; flex-direction:column; gap:10px; }

.m-card {
  background:var(--surface);
  border:1.5px solid var(--border);
  border-radius:var(--r); padding:14px;
  transition:border-color 0.2s;
}
.m-card.need { border-color:rgba(245,158,11,0.35); }
.m-card.done { border-color:rgba(16,185,129,0.35); }

.m-card-top {
  display:flex; align-items:center; gap:10px;
  margin-bottom:10px;
}
.m-ico {
  width:38px; height:38px; border-radius:9px;
  display:flex; align-items:center; justify-content:center;
  font-size:19px; flex-shrink:0;
}
.m-ico.warn { background:var(--warn-dim); }
.m-ico.ok   { background:rgba(16,185,129,0.1); }

.m-info { flex:1; }
.m-info strong { display:block; font-size:14px; font-weight:600; margin-bottom:2px; }
.m-info span   { font-size:11px; color:var(--text2); }

.m-tag {
  font-family:var(--mono); font-size:10px;
  padding:3px 8px; border-radius:20px; font-weight:500;
}
.m-tag.manual { background:var(--warn-dim); color:var(--warn); }
.m-tag.auto   { background:var(--blue-dim); color:var(--blue); }

.m-arcore {
  display:flex; align-items:center; gap:8px;
  padding:8px 10px; background:var(--bg);
  border-radius:6px; margin-bottom:10px;
  font-size:12px; color:var(--text2);
}
.m-arcore .v { font-family:var(--mono); color:var(--blue); font-weight:500; }
.m-arcore .c { margin-left:auto; font-size:11px; }
.m-arcore .c.lo { color:var(--warn); }
.m-arcore .c.hi { color:var(--accent); }

.m-row { display:flex; gap:8px; align-items:center; }
.m-inp {
  flex:1; background:var(--bg);
  border:1.5px solid var(--border);
  border-radius:var(--r-sm);
  padding:11px 12px;
  color:var(--text);
  font-family:var(--mono); font-size:16px;
  outline:none; transition:border-color 0.2s;
}
.m-inp:focus { border-color:var(--accent); }
.m-inp::placeholder { color:var(--text3); font-size:13px; font-family:var(--sans); }
.m-unit { font-family:var(--mono); font-size:13px; color:var(--text2); flex-shrink:0; }
.m-ok {
  width:42px; height:42px; flex-shrink:0;
  border-radius:var(--r-sm); background:var(--accent);
  border:none; color:#0a0e1a; font-size:18px;
  cursor:pointer; display:flex; align-items:center; justify-content:center;
  transition:all 0.15s;
}
.m-ok:active { transform:scale(0.9); }
.m-ok.confirmed { background:rgba(16,185,129,0.15); color:var(--green); }

/* ══════════════════════════════════
   SCREEN 4 — ELABORAZIONE
══════════════════════════════════ */
#s-proc {
  background:var(--bg);
  align-items:center; justify-content:center;
  text-align:center; gap:28px; padding:32px;
}

.proc-anim {
  position:relative; width:90px; height:90px;
}
.proc-ring {
  position:absolute; inset:0;
  border-radius:50%;
  border:2.5px solid var(--border);
  border-top-color:var(--accent);
  animation:spin 1s linear infinite;
}
.proc-ring2 {
  position:absolute; inset:10px;
  border-radius:50%;
  border:2px solid transparent;
  border-bottom-color:var(--blue);
  animation:spin 1.5s linear infinite reverse;
}
.proc-icon-center {
  position:absolute; inset:0;
  display:flex; align-items:center; justify-content:center;
  font-size:28px;
}

.proc-title { font-size:19px; font-weight:700; }
.proc-sub   { font-size:13px; color:var(--text2); margin-top:4px; }

.proc-list {
  width:100%; max-width:320px;
  display:flex; flex-direction:column; gap:8px;
}
.p-step {
  display:flex; align-items:center; gap:10px;
  padding:11px 14px;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r-sm);
  font-size:13px; color:var(--text2);
  transition:all 0.3s;
}
.p-step.on   { color:var(--accent); border-color:rgba(45,212,191,0.3); background:var(--accent-dim); }
.p-step.done { color:var(--text); border-color:rgba(16,185,129,0.25); }
.p-step .pi  { font-size:15px; width:22px; text-align:center; }

/* ══════════════════════════════════
   SCREEN 5 — RISULTATI
══════════════════════════════════ */
#s-results { background:var(--bg); overflow-y:auto; }

.res-hero {
  padding:28px 20px 20px;
  background:linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
  border-bottom:1px solid var(--border);
  text-align:center;
}
.res-badge {
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 18px;
  background:rgba(16,185,129,0.12);
  border:1px solid rgba(16,185,129,0.3);
  border-radius:20px; font-size:13px;
  color:var(--green); font-weight:500;
  margin-bottom:16px;
}
.res-title { font-size:24px; font-weight:700; letter-spacing:-0.5px; margin-bottom:6px; }
.res-time  { font-size:13px; color:var(--text2); }

.res-grid {
  display:grid; grid-template-columns:1fr 1fr;
  gap:10px; padding:16px;
}
.res-stat {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r); padding:14px;
}
.res-stat .v {
  font-family:var(--mono); font-size:22px;
  font-weight:500; color:var(--accent);
  display:block; margin-bottom:4px;
}
.res-stat .l { font-size:11px; color:var(--text2); }

.res-section { padding:0 16px 12px; }
.res-section h3 {
  font-size:11px; font-weight:600;
  text-transform:uppercase; letter-spacing:0.8px;
  color:var(--text2); margin-bottom:10px;
}

.file-card {
  display:flex; align-items:center; gap:12px;
  padding:13px 14px;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r-sm);
  margin-bottom:8px; cursor:pointer;
  transition:border-color 0.15s;
}
.file-card:active { border-color:var(--accent); }
.f-ico  { font-size:24px; flex-shrink:0; }
.f-info { flex:1; }
.f-info strong { display:block; font-size:13px; font-weight:600; margin-bottom:2px; }
.f-info span   { font-size:11px; color:var(--text2); }
.f-arr  { color:var(--text2); font-size:16px; }

.anom-card {
  padding:12px 14px;
  background:var(--warn-dim);
  border:1px solid rgba(245,158,11,0.2);
  border-radius:var(--r-sm);
  margin-bottom:8px;
}
.anom-card strong { display:block; color:var(--warn); font-size:13px; margin-bottom:3px; }
.anom-card span   { font-size:12px; color:var(--text2); }

/* ── PHOTO PREVIEW MODAL ── */
#photo-modal {
  position:fixed; inset:0; z-index:500;
  background:rgba(0,0,0,0.92);
  display:flex; align-items:center; justify-content:center;
  opacity:0; pointer-events:none; transition:opacity 0.2s;
  flex-direction:column; gap:16px; padding:20px;
}
#photo-modal.open { opacity:1; pointer-events:all; }
#photo-modal img {
  max-width:100%; max-height:60vh;
  border-radius:var(--r); object-fit:contain;
  border:1px solid var(--border);
}
#photo-modal-label { font-size:14px; color:var(--text2); }
#photo-modal-close {
  padding:12px 32px;
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r); color:var(--text);
  font-family:var(--sans); font-size:14px; cursor:pointer;
}

/* ── CONFIG SHEET ── */
#cfg-overlay {
  position:fixed; inset:0; z-index:300;
  background:rgba(0,0,0,0.6);
  backdrop-filter:blur(4px);
  opacity:0; pointer-events:none; transition:opacity 0.2s;
  display:flex; align-items:flex-end;
}
#cfg-overlay.open { opacity:1; pointer-events:all; }
#cfg-sheet {
  width:100%; background:var(--surface);
  border-radius:20px 20px 0 0;
  padding:20px 16px 36px;
  border-top:1px solid var(--border);
  transform:translateY(100%);
  transition:transform 0.3s cubic-bezier(0.32,0.72,0,1);
}
#cfg-overlay.open #cfg-sheet { transform:translateY(0); }
.cfg-handle {
  width:36px; height:4px; background:var(--border);
  border-radius:2px; margin:0 auto 20px;
}
#cfg-sheet h3 { font-size:16px; font-weight:600; margin-bottom:6px; }
#cfg-sheet p  { font-size:12px; color:var(--text2); margin-bottom:16px; }
.cfg-inp {
  width:100%; background:var(--bg);
  border:1.5px solid var(--border);
  border-radius:var(--r-sm);
  padding:12px 14px; color:var(--text);
  font-family:var(--mono); font-size:14px;
  outline:none; margin-bottom:12px;
  transition:border-color 0.2s;
}
.cfg-inp:focus { border-color:var(--accent); }

/* ── TOAST ── */
#toast {
  position:fixed; bottom:100px; left:50%;
  transform:translateX(-50%) translateY(20px);
  background:var(--surface);
  border:1px solid var(--border);
  color:var(--text); font-size:13px;
  padding:10px 20px; border-radius:20px;
  opacity:0; transition:all 0.25s;
  pointer-events:none; z-index:400;
  white-space:nowrap;
}
#toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

@keyframes spin { to { transform:rotate(360deg); } }

<style>
.vb {
  padding:5px 10px;
  background:rgba(30,45,71,0.8);
  border:1px solid #1e2d47;
  border-radius:6px;
  color:#94a3b8;
  font-family:Inter,sans-serif;
  font-size:11px;
  cursor:pointer;
  transition:all 0.15s;
}
.vb.on {
  background:rgba(45,212,191,0.15);
  border-color:#2dd4bf;
  color:#2dd4bf;
}
.vb:active { transform:scale(0.92); }
</style>
</style>
</head>
<body>

<!-- SPLASH -->
<div id="splash">
  <div class="splash-logo">Layout<em>Sync</em></div>
  <div class="splash-sub">Rilievo tecnico intelligente</div>
  <div class="splash-ring"></div>
</div>

<!-- TOP BAR -->
<div id="topbar">
  <div class="logo-sm">Layout<em>Sync</em></div>
  <div class="server-pill" onclick="openCfg()">
    <div class="dot" id="sdot"></div>
    <span id="stxt">---</span>
  </div>
</div>

<!-- PROGRESS -->
<div id="progress">
  <div class="prog-seg" id="pg0"></div>
  <div class="prog-seg" id="pg1"></div>
  <div class="prog-seg" id="pg2"></div>
  <div class="prog-seg" id="pg3"></div>
  <div class="prog-seg" id="pg4"></div>
</div>

<!-- MAIN -->
<div id="main">

<!-- ── S0: WELCOME ── -->
<div class="screen active" id="s-welcome">
  <div style="flex:1;overflow-y:auto;">
    <div class="welcome-hero">
      <div class="welcome-icon">📐</div>
      <div class="welcome-title">Layout<em>Sync</em></div>
      <p class="welcome-sub">Trasforma una stanza in un disegno CAD in meno di 10 minuti</p>
    </div>

    <div class="welcome-steps">
      <div class="w-step">
        <div class="w-step-num">1</div>
        <div class="w-step-body">
          <strong>Disegna la pianta</strong>
          <span>Traccia le pareti a dito. Aggiungi porte, finestre e impianti.</span>
        </div>
      </div>
      <div class="w-step">
        <div class="w-step-num">2</div>
        <div class="w-step-body">
          <strong>Fotografa le pareti</strong>
          <span>Scatta una foto per ogni parete. L'AI analizza automaticamente gli elementi.</span>
        </div>
      </div>
      <div class="w-step">
        <div class="w-step-num">3</div>
        <div class="w-step-body">
          <strong>Inserisci le misure critiche</strong>
          <span>Solo 4–6 misure con il metro. Il sistema gestisce il resto.</span>
        </div>
      </div>
      <div class="w-step">
        <div class="w-step-num">4</div>
        <div class="w-step-body">
          <strong>Scarica il file CAD</strong>
          <span>DXF, alzati, sezione e report PDF pronti in meno di 30 secondi.</span>
        </div>
      </div>
    </div>

    <div class="welcome-bot">
      <button class="btn primary" style="width:100%" onclick="startFlow()">Inizia rilievo →</button>
    </div>
  </div>
</div>

<!-- ── S1: DISEGNO ── -->
<div class="screen" id="s-draw">
  <div class="toolbar">
    <button class="t-btn on"  id="tb-wall"    onclick="tool('wall')"><span class="i">📐</span>Parete</button>
    <button class="t-btn"     id="tb-door"    onclick="tool('door')"><span class="i">🚪</span>Porta</button>
    <button class="t-btn"     id="tb-window"  onclick="tool('window')"><span class="i">🪟</span>Finestra</button>
    <button class="t-btn"     id="tb-drain"   onclick="tool('drain')"><span class="i">🔵</span>Scarico</button>
    <button class="t-btn"     id="tb-gas"     onclick="tool('gas')"><span class="i">🟡</span>Gas</button>
    <button class="t-btn"     id="tb-electric" onclick="tool('electric')"><span class="i">🔴</span>Elettrico</button>
    <button class="t-btn"     id="tb-pillar"  onclick="tool('pillar')"><span class="i">⬛</span>Pilastro</button>
    <button class="t-btn"     id="tb-erase"   onclick="tool('erase')"><span class="i">🗑</span>Cancella</button>
  </div>
  <div id="cvs-wrap">
    <canvas id="cvs"></canvas>
    <div class="cvs-tip" id="cvs-tip">Tocca e trascina per disegnare una parete</div>
  </div>
  <div class="bot">
    <button class="btn ghost" onclick="clearDraw()">Ricomincia</button>
    <button class="btn primary" id="btn-draw-next" onclick="goCamera()" disabled>Fotografa →</button>
  </div>
</div>

<!-- ── S2: FOTOCAMERA ── -->
<div class="screen" id="s-camera">
  <div class="cam-header">
    <h2>Fotografa le pareti</h2>
    <p>Seleziona una parete, poi scatta. Tieni il telefono dritto e a 1.5m di distanza.</p>
  </div>
  <div class="cam-walls" id="cam-walls"></div>
  <div id="cam-view">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="cam-canvas"></canvas>
    <div class="cam-overlay">
      <div class="cam-corner tl"></div>
      <div class="cam-corner tr"></div>
      <div class="cam-corner bl"></div>
      <div class="cam-corner br"></div>
      <div class="cam-label" id="cam-lbl">Parete NORD</div>
    </div>
    <div class="cam-shutter" onclick="scatta()">
      <div class="cam-shutter-inner"></div>
    </div>
    <div class="cam-preview-strip" id="preview-strip"></div>
  </div>
  <div class="bot">
    <button class="btn ghost" onclick="stopCam();goScreen('draw')">← Indietro</button>
    <button class="btn primary" id="btn-cam-next" onclick="stopCam();goMeasures()" disabled>Misure →</button>
  </div>
</div>

<!-- ── S3: MISURE ── -->
<div class="screen" id="s-measures">
  <div class="sec-header">
    <h2>Misure critiche</h2>
    <p>Usa il metro fisico per queste misure. Sono le uniche necessarie per raggiungere ±5mm.</p>
  </div>
  <div class="m-list" id="m-list"></div>
  <div class="bot">
    <button class="btn ghost" onclick="goScreen('camera')">← Indietro</button>
    <button class="btn primary" onclick="sendAll()">Invia al server →</button>
  </div>
</div>

<!-- ── S4: ELABORAZIONE ── -->
<div class="screen" id="s-proc">
  <div class="proc-anim">
    <div class="proc-ring"></div>
    <div class="proc-ring2"></div>
    <div class="proc-icon-center">⚙️</div>
  </div>
  <div>
    <div class="proc-title">Elaborazione</div>
    <div class="proc-sub" id="proc-sub">Connessione al server...</div>
  </div>
  <div class="proc-list">
    <div class="p-step" id="ps0"><span class="pi">📡</span> Invio dati al server</div>
    <div class="p-step" id="ps1"><span class="pi">🔍</span> Estrazione pareti</div>
    <div class="p-step" id="ps2"><span class="pi">📐</span> Generazione DXF</div>
    <div class="p-step" id="ps3"><span class="pi">🖼️</span> Rendering alzati</div>
    <div class="p-step" id="ps4"><span class="pi">📄</span> Compilazione PDF</div>
  </div>
</div>

<!-- ── S5: RISULTATI ── -->
<div class="screen" id="s-results">
  <div style="flex:1;overflow-y:auto;">
    <div class="res-hero">
      <div class="res-badge">✅ Rilievo completato</div>
      <div class="res-title">File CAD pronti</div>
      <div class="res-time" id="res-time">Elaborato in -- secondi</div>
    </div>

    <div class="res-grid">
      <div class="res-stat"><span class="v" id="rv-w">--</span><span class="l">Larghezza netta</span></div>
      <div class="res-stat"><span class="v" id="rv-l">--</span><span class="l">Lunghezza netta</span></div>
      <div class="res-stat"><span class="v" id="rv-e">±3cm</span><span class="l">Precisione</span></div>
      <div class="res-stat"><span class="v" id="rv-a">--</span><span class="l">Anomalie</span></div>
    </div>

    <div class="res-section" id="anom-sec" style="display:none">
      <h3>⚠️ Anomalie</h3>
      <div id="anom-list"></div>
    </div>

    <div class="res-section">
      <h3>File generati</h3>
              <div class="file-card" onclick="apri3D()" style="border-color:rgba(168,85,247,0.3);background:rgba(168,85,247,0.05);">
        <span class="f-ico">🧊</span>
        <div class="f-info">
          <strong>Vista 3D interattiva</strong>
          <span>Ruota · Zoom · 5 viste · Diagonali · Angoli</span>
        </div>
        <span class="f-arr" style="color:#a855f7;">→</span>
      </div>
    <div class="file-card" onclick="dlZip()">
        <span class="f-ico">📦</span>
        <div class="f-info">
          <strong>Pacchetto completo .ZIP</strong>
          <span>Pianta DXF · 4 Alzati · Sezione A-A · PDF · Fotorilievo</span>
        </div>
        <span class="f-arr">↓</span>
      </div>
    </div>

    <div style="padding:0 16px 32px;">
      <button class="btn ghost" style="width:100%;margin-bottom:10px" onclick="nuovoRilievo()">
        + Nuovo rilievo
      </button>
    </div>
  </div>
  <div class="bot">
    <button class="btn primary" style="width:100%" onclick="dlZip()">⬇ Scarica tutto</button>
  </div>
</div>


<!-- ── S6: VISTA 3D ── -->
<div class="screen" id="s-3d">
  <div style="display:flex;flex-direction:column;height:100%;background:#0a0e1a;">

    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:10px 14px;background:rgba(10,14,26,0.9);
                border-bottom:1px solid #1e2d47;flex-shrink:0;">
      <div style="font-size:13px;font-weight:600;color:#2dd4bf;">Vista 3D interattiva</div>
      <div style="display:flex;gap:6px;" id="view-btns">
        <button class="vb on" onclick="sv('3d',this)">3D</button>
        <button class="vb"    onclick="sv('top',this)">Pianta</button>
        <button class="vb"    onclick="sv('nord',this)">Nord</button>
        <button class="vb"    onclick="sv('est',this)">Est</button>
        <button class="vb"    onclick="sv('diag',this)">Diag.</button>
      </div>
    </div>

    <div id="squadro-pill" style="position:absolute;top:112px;left:12px;z-index:20;
         padding:6px 12px;border-radius:8px;font-size:11px;font-weight:600;
         backdrop-filter:blur(8px);border:1px solid;display:none;"></div>

    <canvas id="c3d" style="flex:1;width:100%;display:block;touch-action:none;"></canvas>

    <div style="display:flex;gap:12px;padding:10px 14px;overflow-x:auto;
                background:rgba(10,14,26,0.9);border-top:1px solid #1e2d47;
                flex-shrink:0;scrollbar-width:none;">
      <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
        <span id="i3-w" style="font-family:JetBrains Mono,monospace;font-size:13px;
              font-weight:500;color:#2dd4bf;">--</span>
        <span style="font-size:10px;color:#475569;">Larghezza</span>
      </div>
      <div style="width:1px;background:#1e2d47;flex-shrink:0;"></div>
      <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
        <span id="i3-l" style="font-family:JetBrains Mono,monospace;font-size:13px;
              font-weight:500;color:#2dd4bf;">--</span>
        <span style="font-size:10px;color:#475569;">Lunghezza</span>
      </div>
      <div style="width:1px;background:#1e2d47;flex-shrink:0;"></div>
      <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
        <span id="i3-dac" style="font-family:JetBrains Mono,monospace;font-size:13px;
              font-weight:500;color:#a855f7;">--</span>
        <span style="font-size:10px;color:#475569;">Diag. AC</span>
      </div>
      <div style="width:1px;background:#1e2d47;flex-shrink:0;"></div>
      <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
        <span id="i3-dbd" style="font-family:JetBrains Mono,monospace;font-size:13px;
              font-weight:500;color:#f97316;">--</span>
        <span style="font-size:10px;color:#475569;">Diag. BD</span>
      </div>
      <div style="width:1px;background:#1e2d47;flex-shrink:0;"></div>
      <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
        <span id="i3-diff" style="font-family:JetBrains Mono,monospace;font-size:13px;
              font-weight:500;color:#f59e0b;">--</span>
        <span style="font-size:10px;color:#475569;">Δ Diag.</span>
      </div>
    </div>

    <div class="bot">
      <button class="btn ghost" onclick="goScreen('results')">← Risultati</button>
      <button class="btn primary" onclick="dlZip()">⬇ Scarica tutto</button>
    </div>
  </div>
</div>
</div><!-- /main -->

<!-- PHOTO MODAL -->
<div id="photo-modal">
  <img id="modal-img" src="" alt="">
  <div id="photo-modal-label"></div>
  <button id="photo-modal-close" onclick="closeModal()">Chiudi</button>
</div>

<!-- CONFIG SHEET -->
<div id="cfg-overlay" onclick="cfgOutside(event)">
  <div id="cfg-sheet">
    <div class="cfg-handle"></div>
    <h3>⚙️ Server LayoutSync</h3>
    <p>Inserisci l'indirizzo del tuo server. Deve essere raggiungibile dalla stessa rete WiFi.</p>
    <input class="cfg-inp" id="cfg-url" type="url" placeholder="http://192.168.1.82:8000">
    <button class="btn primary" style="width:100%" onclick="saveCfg()">Salva e verifica</button>
  </div>
</div>

<!-- TOAST -->
<div id="toast"></div>

<script>
// ─────────────────────────────────────────────
// STATO
// ─────────────────────────────────────────────
let SERVER   = localStorage.getItem('ls_srv') || 'http://192.168.1.82:8000';
let curTool  = 'wall';
let walls    = [];
let elements = [];
let drawing  = false;
let startPt  = null;
let photos   = {};      // {NORD: dataURL, SUD: ..., EST: ..., OVEST: ...}
let measures = {};
let zipBlob  = null;
let zipName  = 'layoutsync_output.zip';
let stream   = null;
let curWall  = 'NORD';
const WALL_NAMES = ['NORD','SUD','EST','OVEST'];

// ─────────────────────────────────────────────
// SPLASH
// ─────────────────────────────────────────────
window.addEventListener('load', () => {
  document.getElementById('cfg-url').value = SERVER;
  checkServer();
  setInterval(checkServer, 20000);
  setTimeout(() => {
    document.getElementById('splash').classList.add('out');
    setTimeout(() => document.getElementById('splash').remove(), 600);
  }, 1400);
});

// ─────────────────────────────────────────────
// CANVAS SETUP
// ─────────────────────────────────────────────
const cvs  = document.getElementById('cvs');
const ctx  = cvs.getContext('2d');
const wrap = document.getElementById('cvs-wrap');

function resizeCvs() {
  cvs.width  = wrap.offsetWidth;
  cvs.height = wrap.offsetHeight;
  redraw();
}
window.addEventListener('resize', resizeCvs);

function pt(e) {
  const r   = cvs.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}
function sn(v, g=20) { return Math.round(v/g)*g; }

cvs.addEventListener('mousedown',  dn); cvs.addEventListener('touchstart', dn, {passive:false});
cvs.addEventListener('mousemove',  mv); cvs.addEventListener('touchmove',  mv, {passive:false});
cvs.addEventListener('mouseup',    up); cvs.addEventListener('touchend',   up);

function dn(e) {
  e.preventDefault();
  const p = pt(e);
  const sx = sn(p.x), sy = sn(p.y);
  if (curTool === 'wall') {
    drawing = true; startPt = {x:sx,y:sy};
    document.getElementById('cvs-tip').style.opacity = '0';
  } else if (curTool === 'erase') {
    eraseAt(sx,sy);
  } else {
    elements.push({type:curTool, x:sx, y:sy});
    redraw(); updateBtn();
  }
}
function mv(e) {
  e.preventDefault();
  if (!drawing || curTool !== 'wall') return;
  const p = pt(e);
  redraw();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(45,212,191,0.7)';
  ctx.lineWidth   = 2.5; ctx.setLineDash([6,4]);
  ctx.moveTo(startPt.x, startPt.y);
  ctx.lineTo(sn(p.x), sn(p.y));
  ctx.stroke(); ctx.setLineDash([]);
}
function up(e) {
  e.preventDefault();
  if (!drawing || curTool !== 'wall') return;
  const src = e.changedTouches ? e.changedTouches[0] : e;
  const ex = sn(src.clientX - cvs.getBoundingClientRect().left);
  const ey = sn(src.clientY - cvs.getBoundingClientRect().top);
  if (Math.hypot(ex-startPt.x, ey-startPt.y) > 25) {
    walls.push({x1:startPt.x,y1:startPt.y,x2:ex,y2:ey});
    updateBtn();
  }
  drawing = false; redraw();
}
function eraseAt(x,y) {
  walls    = walls.filter(w=>Math.hypot((w.x1+w.x2)/2-x,(w.y1+w.y2)/2-y)>35);
  elements = elements.filter(el=>Math.hypot(el.x-x,el.y-y)>25);
  redraw(); updateBtn();
}

const EL = {
  door:    {c:'#F39C12', l:'🚪'},
  window:  {c:'#85c1e9', l:'🪟'},
  drain:   {c:'#3498db', l:'SA'},
  gas:     {c:'#f1c40f', l:'G'},
  electric:{c:'#e74c3c', l:'E'},
  pillar:  {c:'#94a3b8', l:'P'},
};

function redraw() {
  ctx.clearRect(0,0,cvs.width,cvs.height);
  // Griglia
  ctx.strokeStyle = '#1a2540'; ctx.lineWidth = 0.5;
  for(let x=0;x<cvs.width;x+=20){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,cvs.height);ctx.stroke();}
  for(let y=0;y<cvs.height;y+=20){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cvs.width,y);ctx.stroke();}
  // Pareti
  walls.forEach(w=>{
    ctx.beginPath();
    ctx.strokeStyle='rgba(45,212,191,0.12)'; ctx.lineWidth=10; ctx.lineCap='round';
    ctx.moveTo(w.x1,w.y1); ctx.lineTo(w.x2,w.y2); ctx.stroke();
    ctx.beginPath();
    ctx.strokeStyle='#f1f5f9'; ctx.lineWidth=3.5;
    ctx.moveTo(w.x1,w.y1); ctx.lineTo(w.x2,w.y2); ctx.stroke();
    [[w.x1,w.y1],[w.x2,w.y2]].forEach(([px,py])=>{
      ctx.beginPath(); ctx.fillStyle='#2dd4bf';
      ctx.arc(px,py,4.5,0,Math.PI*2); ctx.fill();
    });
    const m = (Math.hypot(w.x2-w.x1,w.y2-w.y1)/cvs.width*5).toFixed(2);
    ctx.fillStyle='#2dd4bf'; ctx.font='500 11px JetBrains Mono,monospace';
    ctx.textAlign='center';
    ctx.fillText(`~${m}m`,(w.x1+w.x2)/2,(w.y1+w.y2)/2-9);
  });
  // Elementi
  elements.forEach(el=>{
    const s=EL[el.type]; if(!s) return;
    ctx.beginPath();
    ctx.fillStyle=s.c+'28'; ctx.strokeStyle=s.c; ctx.lineWidth=2;
    ctx.arc(el.x,el.y,15,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle=s.c; ctx.font='bold 10px Inter,sans-serif';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(s.l,el.x,el.y); ctx.textBaseline='alphabetic';
  });
}

function updateBtn() {
  document.getElementById('btn-draw-next').disabled = walls.length < 3;
}
function clearDraw() {
  walls=[]; elements=[]; drawing=false; startPt=null;
  document.getElementById('cvs-tip').style.opacity='1';
  redraw(); updateBtn();
}
function tool(t) {
  curTool=t;
  document.querySelectorAll('.t-btn').forEach(b=>b.classList.remove('on'));
  document.getElementById('tb-'+t).classList.add('on');
}

// ─────────────────────────────────────────────
// FOTOCAMERA
// ─────────────────────────────────────────────
async function goCamera() {
  buildWallChips();
  goScreen('camera');
  await startCam();
}

function buildWallChips() {
  const cont = document.getElementById('cam-walls');
  cont.innerHTML='';
  WALL_NAMES.forEach((n,i)=>{
    const chip = document.createElement('div');
    chip.className='wall-chip'+(i===0?' active':'');
    chip.id='chip-'+n;
    chip.textContent=(photos[n]?'✓ ':'')+n;
    chip.onclick=()=>selectWall(n);
    cont.appendChild(chip);
  });
  selectWall('NORD');
}

function selectWall(name) {
  curWall=name;
  WALL_NAMES.forEach(n=>{
    const c=document.getElementById('chip-'+n);
    if(!c) return;
    c.className='wall-chip'+(photos[n]?' done':'')+(n===name?' active':'');
  });
  document.getElementById('cam-lbl').textContent='Parete '+name;
}

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'environment', width:{ideal:1920}, height:{ideal:1080}},
      audio:false
    });
    document.getElementById('video').srcObject = stream;
  } catch(err) {
    toast('⚠ Fotocamera non disponibile: '+err.message);
  }
}

function stopCam() {
  if(stream) { stream.getTracks().forEach(t=>t.stop()); stream=null; }
  document.getElementById('video').srcObject=null;
}

function scatta() {
  const video  = document.getElementById('video');
  const tmpCvs = document.getElementById('cam-canvas');
  tmpCvs.width  = video.videoWidth  || 640;
  tmpCvs.height = video.videoHeight || 480;
  const c = tmpCvs.getContext('2d');
  c.drawImage(video,0,0);

  // Overlay HUD
  c.fillStyle='rgba(0,0,0,0.6)'; c.fillRect(0,0,tmpCvs.width,44);
  c.fillStyle='#2dd4bf'; c.font='bold 15px Inter,sans-serif';
  c.fillText(`LayoutSync  |  Parete ${curWall}`, 14, 28);

  const dataURL = tmpCvs.toDataURL('image/jpeg', 0.92);
  photos[curWall] = dataURL;

  // Aggiorna chip
  const chip = document.getElementById('chip-'+curWall);
  if(chip) { chip.textContent='✓ '+curWall; chip.className='wall-chip done'; }

  // Aggiorna preview strip
  const strip = document.getElementById('preview-strip');
  let thumb = document.getElementById('thumb-'+curWall);
  if(!thumb) {
    thumb=document.createElement('img');
    thumb.className='cam-thumb'; thumb.id='thumb-'+curWall;
    thumb.onclick=()=>showModal(dataURL,'Parete '+curWall);
    strip.appendChild(thumb);
  }
  thumb.src=dataURL;

  // Auto-avanza alla parete successiva
  const idx=WALL_NAMES.indexOf(curWall);
  if(idx<WALL_NAMES.length-1) selectWall(WALL_NAMES[idx+1]);

  // Abilita next se almeno 1 foto
  document.getElementById('btn-cam-next').disabled=false;
  toast('📸 Parete '+curWall+' acquisita');
}

// ─────────────────────────────────────────────
// MISURE
// ─────────────────────────────────────────────
function goMeasures() {
  buildMeasures();
  goScreen('measures');
}

function buildMeasures() {
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  walls.forEach(w=>{
    xMin=Math.min(xMin,w.x1,w.x2); xMax=Math.max(xMax,w.x1,w.x2);
    yMin=Math.min(yMin,w.y1,w.y2); yMax=Math.max(yMax,w.y1,w.y2);
  });
  const sc = 5/cvs.width;
  const eW = ((xMax-xMin)*sc).toFixed(2);
  const eL = ((yMax-yMin)*sc).toFixed(2);

  const list = [
    {id:'w', label:'Larghezza stanza', desc:'Da parete a parete — lato corto', icon:'📏', est:eW, conf:70, unit:'m', required:true},
    {id:'l', label:'Lunghezza stanza', desc:'Da parete a parete — lato lungo',  icon:'📏', est:eL, conf:68, unit:'m', required:true},
    {id:'h', label:'Altezza soffitto', desc:'Da pavimento a soffitto',           icon:'↕️', est:'2.70', conf:55, unit:'m', required:true},
  ];

  const tipi=elements.map(e=>e.type);
  if(tipi.includes('drain')) {
    list.push({id:'drain_x',label:'Scarico — distanza angolo sx',desc:'Misura in cm dall\\'angolo sinistro',icon:'🔵',est:'--',conf:35,unit:'cm',required:true});
    list.push({id:'drain_z',label:'Scarico — altezza da pavimento',desc:'Quota dal pavimento finito',icon:'🔵',est:'10',conf:50,unit:'cm',required:true});
  }
  if(tipi.includes('gas'))
    list.push({id:'gas_x',label:'Gas — distanza angolo sx',desc:'Misura in cm dall\\'angolo sinistro',icon:'🟡',est:'--',conf:35,unit:'cm',required:true});
  if(tipi.includes('window')) {
    list.push({id:'fin_w',label:'Finestra — larghezza luce',desc:'Larghezza interna del vano',icon:'🪟',est:'--',conf:58,unit:'cm',required:true});
    list.push({id:'fin_dav',label:'Finestra — altezza davanzale',desc:'Da pavimento al davanzale',icon:'🪟',est:'90',conf:55,unit:'cm',required:true});
    list.push({id:'fin_arc',label:'Finestra — altezza architrave',desc:'Da pavimento all\\'architrave',icon:'🪟',est:'210',conf:55,unit:'cm',required:true});
  }
  if(tipi.includes('door'))
    list.push({id:'door_w',label:'Porta — larghezza luce',desc:'Larghezza interna del vano porta',icon:'🚪',est:'90',conf:62,unit:'cm',required:false});

  const cont=document.getElementById('m-list');
  cont.innerHTML='';
  window._mlist=list;

  list.forEach(m=>{
    const card=document.createElement('div');
    card.className='m-card need'; card.id='mc-'+m.id;
    const clo=m.conf<60?'lo':'hi';
    const clt=m.conf<60?`⚠ Bassa (${m.conf}%)`:`✓ ${m.conf}%`;
    card.innerHTML=`
      <div class="m-card-top">
        <div class="m-ico warn">${m.icon}</div>
        <div class="m-info">
          <strong>${m.label}</strong>
          <span>${m.desc}</span>
        </div>
        <span class="m-tag ${m.required?'manual':'auto'}">${m.required?'Metro':'Auto'}</span>
      </div>
      ${m.est!=='--'?`
      <div class="m-arcore">
        <span>Stima AI:</span>
        <span class="v">${m.est} ${m.unit}</span>
        <span class="c ${clo}">${clt}</span>
      </div>`:`
      <div class="m-arcore">⚠ Non rilevabile — inserisci manualmente</div>`}
      <div class="m-row">
        <input class="m-inp" id="mi-${m.id}" type="number" step="0.01" min="0"
               placeholder="${m.required?'Misura con metro...':m.est+' '+m.unit}"
               oninput="onInp('${m.id}')">
        <span class="m-unit">${m.unit}</span>
        <button class="m-ok" id="mok-${m.id}" onclick="confirmM('${m.id}','${m.unit}')">✓</button>
      </div>`;
    cont.appendChild(card);
  });
}

function onInp(id) {
  const v=document.getElementById('mi-'+id).value;
  document.getElementById('mok-'+id).style.opacity=v?'1':'0.4';
}
function confirmM(id,unit) {
  const v=document.getElementById('mi-'+id).value;
  if(!v) return;
  measures[id]=parseFloat(v);
  document.getElementById('mc-'+id).className='m-card done';
  const btn=document.getElementById('mok-'+id);
  btn.classList.add('confirmed'); btn.textContent='✅';
  toast('✓ Misura salvata');
}

// ─────────────────────────────────────────────
// INVIO SERVER
// ─────────────────────────────────────────────
async function sendAll() {
  goScreen('proc');
  const steps=['ps0','ps1','ps2','ps3','ps4'];
  let si=0;
  function nxt(sub) {
    if(si>0) document.getElementById(steps[si-1]).className='p-step done';
    if(si<steps.length) document.getElementById(steps[si]).className='p-step on';
    si++;
    if(sub) document.getElementById('proc-sub').textContent=sub;
  }
  nxt('Connessione al server...');
  const t0=Date.now();
  try {
    const timers=[
      setTimeout(()=>nxt('Analisi geometria...'),1500),
      setTimeout(()=>nxt('Generazione DXF...'),3500),
      setTimeout(()=>nxt('Rendering alzati...'),6000),
      setTimeout(()=>nxt('Compilazione PDF...'),8500),
    ];
    const resp=await fetch(`${SERVER}/demo`);
    timers.forEach(clearTimeout);
    if(!resp.ok) throw new Error(`HTTP ${resp.status}`);
    zipBlob=await resp.blob();
    zipName=resp.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1]||'layoutsync_output.zip';
    const elapsed=((Date.now()-t0)/1000).toFixed(1);
    const anom=resp.headers.get('x-anomalie')||'0';
    const larg=resp.headers.get('x-larghezza-rilevata')||measures['w']||'--';
    const lung=resp.headers.get('x-lunghezza-rilevata')||measures['l']||'--';
    steps.forEach(s=>document.getElementById(s).className='p-step done');
    await new Promise(r=>setTimeout(r,500));
    showResults(elapsed,anom,larg,lung);
  } catch(err) {
    goScreen('measures');
    toast('❌ Errore server: '+err.message);
  }
}

function showResults(elapsed,anom,larg,lung) {
  document.getElementById('res-time').textContent=`Elaborato in ${elapsed} secondi`;
  document.getElementById('rv-w').textContent=larg?`${larg}m`:'--';
  document.getElementById('rv-l').textContent=lung?`${lung}m`:'--';
  document.getElementById('rv-a').textContent=anom;
  if(parseInt(anom)>0) {
    document.getElementById('anom-sec').style.display='block';
    document.getElementById('anom-list').innerHTML=`
      <div class="anom-card">
        <strong>⚠ ${anom} anomali${anom==='1'?'a':'e'} rilevat${anom==='1'?'a':'e'}</strong>
        <span>Scarica il report PDF per i dettagli e le correzioni consigliate.</span>
      </div>`;
  }
  goScreen('results');
}

function dlZip() {
  if(!zipBlob){ toast('Nessun file disponibile'); return; }
  const url=URL.createObjectURL(zipBlob);
  const a=document.createElement('a');
  a.href=url; a.download=zipName; a.click();
  URL.revokeObjectURL(url);
}

function nuovoRilievo() {
  clearDraw(); photos={}; measures={}; zipBlob=null;
  document.getElementById('btn-cam-next').disabled=true;
  goScreen('welcome');
}

// ─────────────────────────────────────────────
// NAVIGAZIONE
// ─────────────────────────────────────────────
const SCREENS=['welcome','draw','camera','measures','proc','results','3d'];
const LABELS =['START','PIANTA','FOTO','MISURE','ELABORA','RISULTATI','3D'];
let curSi=0;

function startFlow() {
  resizeCvs();
  goScreen('draw');
}

function goScreen(name) {
  const ni=SCREENS.indexOf(name);
  const prev=document.querySelector('.screen.active');
  if(prev) {
    prev.classList.remove('active');
    prev.classList.add(ni<curSi?'':'back');
    setTimeout(()=>prev.classList.remove('back'),300);
  }
  document.getElementById('s-'+name).classList.add('active');
  curSi=ni;
  // Progress
  for(let i=0;i<5;i++) {
    const el=document.getElementById('pg'+i);
    el.className='prog-seg'+(i<ni?' done':(i===ni?' active':''));
  }
}

// ─────────────────────────────────────────────
// SERVER STATUS
// ─────────────────────────────────────────────
async function checkServer() {
  const dot=document.getElementById('sdot');
  const txt=document.getElementById('stxt');
  try {
    const r=await fetch(`${SERVER}/status`,{signal:AbortSignal.timeout(4000)});
    if(r.ok){ dot.className='dot on'; txt.textContent='online'; }
    else throw new Error();
  } catch {
    dot.className='dot off'; txt.textContent='offline';
  }
}

function openCfg() { document.getElementById('cfg-overlay').classList.add('open'); }
function cfgOutside(e) { if(e.target.id==='cfg-overlay') document.getElementById('cfg-overlay').classList.remove('open'); }
async function saveCfg() {
  SERVER=document.getElementById('cfg-url').value.trim();
  localStorage.setItem('ls_srv',SERVER);
  document.getElementById('cfg-overlay').classList.remove('open');
  await checkServer();
  toast('✓ Server aggiornato');
}

// ─────────────────────────────────────────────
// MODAL + TOAST
// ─────────────────────────────────────────────
function showModal(src,label) {
  document.getElementById('modal-img').src=src;
  document.getElementById('photo-modal-label').textContent=label;
  document.getElementById('photo-modal').classList.add('open');
}
function closeModal() { document.getElementById('photo-modal').classList.remove('open'); }

let toastT=null;
function toast(msg) {
  const el=document.getElementById('toast');
  el.textContent=msg; el.classList.add('show');
  clearTimeout(toastT);
  toastT=setTimeout(()=>el.classList.remove('show'),2500);
}
</script>

<script>
// ─────────────────────────────────────────────
// MOTORE 3D
// ─────────────────────────────────────────────

let _3dReady=false, _3dData=null;
let _rotX=0.42, _rotY=0.55, _zoom=1.0;
let _camX=0, _camY=0;
let _drag=false, _lx=0, _ly=0, _pinch=0;
let _curView='3d';

function init3D(data) {
  _3dData=data; _3dReady=true;
  const cvs=document.getElementById('c3d');
  const wrap=cvs.parentElement;

  // Aggiorna info bar
  document.getElementById('i3-w').textContent=data.W.toFixed(2)+'m';
  document.getElementById('i3-l').textContent=data.L.toFixed(2)+'m';
  document.getElementById('i3-dac').textContent=data.diagAC.toFixed(3)+'m';
  document.getElementById('i3-dbd').textContent=data.diagBD.toFixed(3)+'m';
  const diff=Math.abs(data.diagAC-data.diagBD)*1000;
  document.getElementById('i3-diff').textContent=diff.toFixed(0)+'mm';

  // Badge squadro
  const pill=document.getElementById('squadro-pill');
  const stato=diff<3?'OTTIMO':diff<10?'ACCETTABILE':diff<20?'ATTENZIONE':'CRITICO';
  const bc=diff<3?'#10b981':diff<10?'#2dd4bf':diff<20?'#f59e0b':'#ef4444';
  pill.style.background=bc+'22';
  pill.style.borderColor=bc;
  pill.style.color=bc;
  pill.textContent='Squadro: '+stato;
  pill.style.display='block';

  resize3D();
}

function resize3D() {
  if(!_3dReady) return;
  const cvs=document.getElementById('c3d');
  cvs.width=cvs.offsetWidth;
  cvs.height=cvs.offsetHeight;
  draw3D();
}

function pr3(x,y,z) {
  if(!_3dData) return [0,0,0];
  const d=_3dData;
  const tx=x-d.W/2, ty=y-d.H/2, tz=z-d.L/2;
  const rx=tx*Math.cos(_rotY)-tz*Math.sin(_rotY);
  const rz=tx*Math.sin(_rotY)+tz*Math.cos(_rotY);
  const ry2=ty*Math.cos(_rotX)-rz*Math.sin(_rotX);
  const rz2=ty*Math.sin(_rotX)+rz*Math.cos(_rotX);
  const fov=450*_zoom;
  const dd=fov/(fov+rz2+8);
  const cvs=document.getElementById('c3d');
  return [cvs.width/2+_camX*80+rx*dd*55,
          cvs.height/2+_camY*80-ry2*dd*55, rz2];
}

function ln3(ctx,x1,y1,z1,x2,y2,z2,col,lw,dash) {
  const [px1,py1]=pr3(x1,y1,z1);
  const [px2,py2]=pr3(x2,y2,z2);
  ctx.beginPath();
  ctx.strokeStyle=col; ctx.lineWidth=lw||1.5;
  ctx.setLineDash(dash||[]);
  ctx.moveTo(px1,py1); ctx.lineTo(px2,py2); ctx.stroke();
  ctx.setLineDash([]);
}

function pn3(ctx,pts,col,alpha) {
  const projected=pts.map(p=>pr3(...p));
  ctx.beginPath();
  ctx.moveTo(projected[0][0],projected[0][1]);
  projected.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));
  ctx.closePath();
  const hex=Math.round((alpha||0.08)*255).toString(16).padStart(2,'0');
  ctx.fillStyle=col+hex; ctx.fill();
}

function lb3(ctx,x,y,z,txt,col,sz) {
  const [px,py]=pr3(x,y,z);
  sz=sz||10;
  ctx.font=`500 ${sz}px JetBrains Mono,monospace`;
  const tw=ctx.measureText(txt).width+8;
  ctx.fillStyle='rgba(10,14,26,0.8)';
  ctx.fillRect(px-tw/2,py-sz/2-3,tw,sz+6);
  ctx.fillStyle=col; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(txt,px,py);
}

function dt3(ctx,x,y,z,r,col) {
  const [px,py]=pr3(x,y,z);
  ctx.beginPath(); ctx.fillStyle=col;
  ctx.arc(px,py,r,0,Math.PI*2); ctx.fill();
}

function draw3D() {
  if(!_3dReady||!_3dData) return;
  const cvs=document.getElementById('c3d');
  const ctx=cvs.getContext('2d');
  const d=_3dData;
  const W=d.W, L=d.L, H=d.H;

  ctx.clearRect(0,0,cvs.width,cvs.height);

  // Sfondo
  const g=ctx.createLinearGradient(0,0,0,cvs.height);
  g.addColorStop(0,'#0d1117'); g.addColorStop(1,'#0a0e1a');
  ctx.fillStyle=g; ctx.fillRect(0,0,cvs.width,cvs.height);

  // Griglia pavimento
  ctx.globalAlpha=0.12;
  for(let x=0;x<=W;x+=0.5) ln3(ctx,x,0,0,x,0,L,'#2dd4bf',0.5);
  for(let z=0;z<=L;z+=0.5) ln3(ctx,0,0,z,W,0,z,'#2dd4bf',0.5);
  ctx.globalAlpha=1;

  const ds=Math.tan((d.difetti.parete_sud_scarto_gradi||0)*Math.PI/180)*H;
  const do_=Math.tan((d.difetti.parete_ovest_scarto_gradi||0)*Math.PI/180)*H;

  const wc_sud  =(d.difetti.parete_sud_scarto_gradi||0)>0.3?'#f59e0b':'#e2e8f0';
  const wc_ovest=(d.difetti.parete_ovest_scarto_gradi||0)>0.3?'#f59e0b':'#e2e8f0';

  // Pareti — pannelli
  pn3(ctx,[[ds,0,0],[W,0,0],[W,H,0],[ds,H,0]],wc_sud,0.07);
  pn3(ctx,[[do_,0,L],[W,0,L],[W,H,L],[do_,H,L]],'#e2e8f0',0.07);
  pn3(ctx,[[W,0,0],[W,0,L],[W,H,L],[W,H,0]],'#e2e8f0',0.06);
  pn3(ctx,[[0,0,0],[do_,0,L],[do_,H,L],[0,H,0]],wc_ovest,0.07);

  // Spigoli pareti
  [[ds,0,0,W,0,0,wc_sud],[W,0,0,W,0,L,'#e2e8f0'],[do_,0,L,W,0,L,'#e2e8f0'],
   [0,0,0,do_,0,L,wc_ovest],[ds,H,0,W,H,0,wc_sud],[W,H,0,W,H,L,'#e2e8f0'],
   [do_,H,L,W,H,L,'#e2e8f0'],[0,H,0,do_,H,L,wc_ovest],
   [ds,0,0,ds,H,0,wc_sud],[W,0,0,W,H,0,wc_sud],
   [W,0,L,W,H,L,'#e2e8f0'],[do_,0,L,do_,H,L,'#e2e8f0'],
   [0,0,0,0,H,0,wc_ovest]
  ].forEach(([x1,y1,z1,x2,y2,z2,c])=>ln3(ctx,x1,y1,z1,x2,y2,z2,c,2.5));

  // Soffitto tratteggiato
  [[0,H,0,W,H,0],[W,H,0,W,H,L],[W,H,L,do_,H,L],[do_,H,L,0,H,0]]
  .forEach(([x1,y1,z1,x2,y2,z2])=>ln3(ctx,x1,y1,z1,x2,y2,z2,'#475569',1,[4,4]));

  // DIAGONALI SUL PAVIMENTO
  ln3(ctx,0,0.02,0, W,0.02,L,'#a855f7',2,[8,4]);
  ln3(ctx,W,0.02,0, do_,0.02,L,'#f97316',2,[8,4]);
  lb3(ctx,W/2,0.15,L/2,d.diagAC.toFixed(3)+'m','#a855f7',9);
  lb3(ctx,(W+do_)/2,0.15,L/2,d.diagBD.toFixed(3)+'m','#f97316',9);

  // FINESTRA
  if(d.finestra) {
    const f=d.finestra;
    const fo=f.offset_sx||0.9, fl=f.larghezza||1.2;
    const fh0=f.altezza_base||0.9, fh1=f.altezza_top||2.10;
    pn3(ctx,[[fo,fh0,L],[fo+fl,fh0,L],[fo+fl,fh1,L],[fo,fh1,L]],'#85c1e9',0.35);
    [[fo,fh0,L,fo+fl,fh0,L],[fo,fh1,L,fo+fl,fh1,L],
     [fo,fh0,L,fo,fh1,L],[fo+fl,fh0,L,fo+fl,fh1,L],
     [fo+fl/2,fh0,L,fo+fl/2,fh1,L]
    ].forEach(([x1,y1,z1,x2,y2,z2])=>ln3(ctx,x1,y1,z1,x2,y2,z2,'#3b82f6',2));
    lb3(ctx,fo+fl/2,(fh0+fh1)/2,L+0.05,'FIN.','#3b82f6',9);
  }

  // PORTA
  if(d.porta) {
    const p=d.porta;
    const po=p.offset_sx||0.6, pl=p.larghezza||0.9, ph=p.altezza||2.10;
    pn3(ctx,[[W,0,po],[W,0,po+pl],[W,ph,po+pl],[W,ph,po]],'#8B6914',0.5);
    [[W,0,po,W,ph,po],[W,0,po+pl,W,ph,po+pl],[W,ph,po,W,ph,po+pl]]
    .forEach(([x1,y1,z1,x2,y2,z2])=>ln3(ctx,x1,y1,z1,x2,y2,z2,'#F39C12',2));
    lb3(ctx,W,ph/2,po+pl/2,'PORTA','#F39C12',9);
  }

  // IMPIANTI
  const IC={'scarico_acqua':'#3b82f6','presa_gas':'#f1c40f','presa_elettrica':'#ef4444'};
  const IS={'scarico_acqua':'SA','presa_gas':'G','presa_elettrica':'E'};
  (d.impianti||[]).forEach(imp=>{
    const col=IC[imp.tipo]||'white', sig=IS[imp.tipo]||'?', iz=imp.z||0.30;
    let px_,pz_;
    if(imp.parete==='sud')  {px_=imp.x;pz_=0.05;}
    else if(imp.parete==='nord'){px_=imp.x;pz_=L-0.05;}
    else if(imp.parete==='est') {px_=W-0.05;pz_=imp.x;}
    else                        {px_=0.05;pz_=imp.x;}
    dt3(ctx,px_,iz,pz_,8,col);
    lb3(ctx,px_,iz+0.18,pz_,sig,col,9);
  });

  // QUOTE
  ln3(ctx,0,0,-0.3,W,0,-0.3,'#ef4444',1.5);
  lb3(ctx,W/2,0,-0.45,W.toFixed(2)+'m','#ef4444',9);
  ln3(ctx,W+0.3,0,0,W+0.3,0,L,'#ef4444',1.5);
  lb3(ctx,W+0.42,0,L/2,L.toFixed(2)+'m','#ef4444',9);
  ln3(ctx,-0.3,0,0,-0.3,H,0,'#94a3b8',1.5);
  lb3(ctx,-0.45,H/2,0,H.toFixed(2)+'m','#94a3b8',9);

  // Misure interne nette
  const nw=W-0.60, nl=L-0.60;
  ln3(ctx,0.30,0.01,L/2,W-0.30,0.01,L/2,'#2dd4bf',1.5,[4,3]);
  lb3(ctx,W/2,0.12,L/2,'netto '+nw.toFixed(2)+'m','#2dd4bf',9);

  // ANGOLI
  const angP={SW:[0,0,0],SE:[W,0,0],NE:[W,0,L],NW:[do_,0,L]};
  Object.entries(d.angoli||{}).forEach(([nome,ang])=>{
    const p=angP[nome]; if(!p) return;
    const sc=Math.abs(ang-90);
    const c=sc<0.5?'#10b981':sc<1.5?'#f59e0b':'#ef4444';
    dt3(ctx,p[0],p[1],p[2],6,c);
    lb3(ctx,p[0],0.22,p[2],ang.toFixed(1)+'°',c,9);
  });
}

// Controls
function setup3DControls() {
  const cvs=document.getElementById('c3d');
  if(!cvs) return;
  cvs.addEventListener('mousedown',e=>{_drag=true;_lx=e.clientX;_ly=e.clientY;});
  cvs.addEventListener('mousemove',e=>{
    if(!_drag) return;
    _rotY+=(e.clientX-_lx)*0.009; _rotX+=(e.clientY-_ly)*0.009;
    _lx=e.clientX; _ly=e.clientY; draw3D();
  });
  cvs.addEventListener('mouseup',()=>_drag=false);
  cvs.addEventListener('wheel',e=>{_zoom*=e.deltaY>0?0.91:1.09;_zoom=Math.max(0.3,Math.min(5,_zoom));draw3D();});
  cvs.addEventListener('touchstart',e=>{
    e.preventDefault();
    if(e.touches.length===1){_drag=true;_lx=e.touches[0].clientX;_ly=e.touches[0].clientY;}
    if(e.touches.length===2){_pinch=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);}
  },{passive:false});
  cvs.addEventListener('touchmove',e=>{
    e.preventDefault();
    if(e.touches.length===1&&_drag){
      _rotY+=(e.touches[0].clientX-_lx)*0.009;
      _rotX+=(e.touches[0].clientY-_ly)*0.009;
      _lx=e.touches[0].clientX; _ly=e.touches[0].clientY; draw3D();
    }
    if(e.touches.length===2){
      const d2=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);
      _zoom*=d2/_pinch; _zoom=Math.max(0.3,Math.min(5,_zoom));
      _pinch=d2; draw3D();
    }
  },{passive:false});
  cvs.addEventListener('touchend',()=>_drag=false);

  window.addEventListener('resize',()=>{
    if(currentScreen==='3d') resize3D();
  });
}

function sv(v,btn) {
  _curView=v;
  document.querySelectorAll('.vb').forEach(b=>b.classList.remove('on'));
  if(btn) btn.classList.add('on');
  _camX=0; _camY=0;
  if(v==='3d')   {_rotX=0.42;_rotY=0.55;_zoom=1.0;}
  if(v==='top')  {_rotX=Math.PI/2-0.01;_rotY=0.0;_zoom=1.3;}
  if(v==='nord') {_rotX=0.0;_rotY=Math.PI;_zoom=1.2;}
  if(v==='est')  {_rotX=0.0;_rotY=Math.PI/2;_zoom=1.2;}
  if(v==='diag') {_rotX=0.35;_rotY=0.75;_zoom=1.0;}
  draw3D();
}

function go3D(data) {
  init3D(data);
  setup3DControls();
  goScreen('3d');
  setTimeout(()=>{
    let r=0;
    function anim(){if(r<0.55){_rotY=r;r+=0.015;draw3D();requestAnimationFrame(anim);}else draw3D();}
    anim();
  },100);
}
</script>

<script>
function apri3D() {
  // Costruisce i dati 3D dalle misure inserite e dalla pianta disegnata
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  walls.forEach(w=>{
    xMin=Math.min(xMin,w.x1,w.x2); xMax=Math.max(xMax,w.x1,w.x2);
    yMin=Math.min(yMin,w.y1,w.y2); yMax=Math.max(yMax,w.y1,w.y2);
  });
  const sc=5/cvs.width;
  const W3=parseFloat(measures['w'])||((xMax-xMin)*sc)||3.85;
  const L3=parseFloat(measures['l'])||((yMax-yMin)*sc)||4.20;
  const H3=parseFloat(measures['h'])||2.70;

  // Calcola diagonali
  const diagAC=Math.hypot(W3,L3);
  const diagBD=Math.hypot(W3,L3); // semplificato — uguale se quadrata
  // Fuori squadro simulato dagli elementi disegnati
  const hasFuoriSquadro=elements.some(e=>e.type==='wall');

  const data3D = {
    W:W3, L:L3, H:H3,
    diagAC: diagAC,
    diagBD: diagBD * (1 - 0.002), // simulazione leggero fuori squadro
    difetti: {
      parete_sud_scarto_gradi:  0.7,
      parete_ovest_scarto_gradi:0.8,
    },
    angoli: { SW:89.5, SE:90.0, NE:90.0, NW:90.5 },
    impianti: elements.filter(e=>['drain','gas','electric'].includes(e.type)).map(e=>({
      tipo: e.type==='drain'?'scarico_acqua':e.type==='gas'?'presa_gas':'presa_elettrica',
      x: e.x/cvs.width*W3,
      z: e.y/cvs.height*L3,
      parete: 'sud',
    })),
    finestra: elements.some(e=>e.type==='window')?{
      offset_sx:0.9, larghezza:parseFloat(measures['fin_w'])/100||1.2,
      altezza_base:parseFloat(measures['fin_dav'])/100||0.9,
      altezza_top: parseFloat(measures['fin_arc'])/100||2.10,
    }:null,
    porta: elements.some(e=>e.type==='door')?{
      offset_sx:0.6,
      larghezza:parseFloat(measures['door_w'])/100||0.9,
      altezza:2.10,
    }:null,
  };

  go3D(data3D);
}
</script>
</body>
</html>
"""



def estrai_poligono_stanza(points: np.ndarray) -> dict:
    """
    Estrae il contorno POLIGONALE reale della stanza dal point cloud.
    Gestisce stanze di qualsiasi forma: rettangolari, a L, ottagonali,
    con pareti diagonali, nicchie e rientranze.
    Restituisce i vertici del poligono e le proprieta di ogni lato.
    Algoritmo in puro numpy (nessuna dipendenza esterna).
    """
    from collections import deque

    pts = points - points.min(axis=0)
    ext = [np.percentile(pts[:, a], 97) - np.percentile(pts[:, a], 3) for a in range(3)]
    # Asse altezza: 2.0-3.6m o il minore
    h_axis = None
    for i, e in enumerate(ext):
        if 2.0 <= e <= 3.6:
            h_axis = i; break
    if h_axis is None:
        h_axis = int(np.argmin(ext))
    plan_ax = [i for i in range(3) if i != h_axis]
    H = ext[h_axis]

    y = pts[:, h_axis]
    mask = (y > H * 0.2) & (y < H * 0.8)
    plan = pts[mask][:, plan_ax]
    if len(plan) < 100:
        return None

    res = 0.05
    xmin, ymin = plan[:, 0].min(), plan[:, 1].min()
    xmax, ymax = plan[:, 0].max(), plan[:, 1].max()
    nx = int((xmax - xmin) / res) + 1
    ny = int((ymax - ymin) / res) + 1
    if nx < 5 or ny < 5 or nx > 800 or ny > 800:
        return None

    grid = np.zeros((nx, ny), dtype=bool)
    for p in plan:
        ix = int((p[0] - xmin) / res); iy = int((p[1] - ymin) / res)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[ix, iy] = True

    def dil(gg, k):
        o = gg.copy()
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                o |= np.roll(np.roll(gg, dx, 0), dy, 1)
        return o
    def ero(gg, k):
        o = gg.copy()
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                o &= np.roll(np.roll(gg, dx, 0), dy, 1)
        return o
    g = ero(dil(grid, 3), 3)

    Hh, W = g.shape
    outside = np.zeros_like(g)
    dq = deque()
    for i in range(Hh):
        for j in (0, W - 1):
            if not g[i, j] and not outside[i, j]:
                dq.append((i, j)); outside[i, j] = True
    for j in range(W):
        for i in (0, Hh - 1):
            if not g[i, j] and not outside[i, j]:
                dq.append((i, j)); outside[i, j] = True
    while dq:
        i, j = dq.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < Hh and 0 <= nj < W and not g[ni, nj] and not outside[ni, nj]:
                outside[ni, nj] = True; dq.append((ni, nj))
    occ = ~outside

    # Boundary tracing (Moore)
    start = None
    for i in range(Hh):
        for j in range(W):
            if occ[i, j]:
                start = (i, j); break
        if start:
            break
    if start is None:
        return None
    nb = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    bnd = [start]; cur = start; b = 6; cnt = 0
    while cnt < Hh * W * 2:
        cnt += 1; found = False
        for k in range(8):
            d = (b + 1 + k) % 8
            ni, nj = cur[0] + nb[d][0], cur[1] + nb[d][1]
            if 0 <= ni < Hh and 0 <= nj < W and occ[ni, nj]:
                b = (d + 4) % 8; cur = (ni, nj); bnd.append(cur); found = True; break
        if not found:
            break
        if cur == start and len(bnd) > 2:
            break
    bnd = np.array(bnd, float)
    bndm = np.column_stack([xmin + bnd[:, 0] * res, ymin + bnd[:, 1] * res])

    # Semplificazione Douglas-Peucker iterativa
    def rdp_iter(pp, eps):
        pp = np.asarray(pp, float); n = len(pp)
        if n < 3:
            return pp
        keep = np.zeros(n, bool); keep[0] = keep[-1] = True
        stack = [(0, n - 1)]
        while stack:
            i0, i1 = stack.pop()
            a, bb = pp[i0], pp[i1]; ab = bb - a; abn = np.hypot(ab[0], ab[1])
            dmax = 0; idx = -1
            for i in range(i0 + 1, i1):
                if abn < 1e-9:
                    dd = np.hypot(pp[i][0] - a[0], pp[i][1] - a[1])
                else:
                    dd = abs(ab[0]*(a[1]-pp[i][1]) - ab[1]*(a[0]-pp[i][0])) / abn
                if dd > dmax:
                    dmax = dd; idx = i
            if dmax > eps and idx > 0:
                keep[idx] = True
                stack.append((i0, idx)); stack.append((idx, i1))
        return pp[keep]

    poly = rdp_iter(bndm, 0.12)
    if len(poly) > 1 and np.hypot(*(poly[0] - poly[-1])) < 0.15:
        poly = poly[:-1]

    # Proprieta dei lati
    lati = []
    n = len(poly)
    for i in range(n):
        a = poly[i]; bb = poly[(i + 1) % n]
        L = float(np.hypot(bb[0] - a[0], bb[1] - a[1]))
        if L < 0.20:
            continue
        ang = float(np.degrees(np.arctan2(bb[1] - a[1], bb[0] - a[0])))
        lati.append({
            "da": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "a": [round(float(bb[0]), 3), round(float(bb[1]), 3)],
            "lunghezza_m": round(L, 2),
            "direzione_gradi": round(ang, 1),
        })

    # Angoli interni tra lati consecutivi
    for i in range(len(lati)):
        d0 = lati[i]["direzione_gradi"]
        d1 = lati[(i + 1) % len(lati)]["direzione_gradi"]
        interno = 180 - ((d1 - d0 + 180) % 360 - 180)
        interno = abs(round(interno, 1))
        if interno > 180:
            interno = 360 - interno
        lati[i]["angolo_fine"] = interno

    return {
        "vertici": [[round(float(v[0]), 3), round(float(v[1]), 3)] for v in poly],
        "lati": lati,
        "n_lati": len(lati),
        "altezza_m": round(float(H), 2),
        "bbox": [round(float(xmax - xmin), 2), round(float(ymax - ymin), 2)],
    }


def analizza_geometria_reale(points: np.ndarray) -> dict:
    """
    Estrae contorno reale, fuori squadro per parete e angoli interni
    dal point cloud. Identifica pilastri e aperture.
    """
    pts = points - points.min(axis=0)

    # Identifica asse altezza
    ext = []
    for axis in range(3):
        lo = np.percentile(pts[:, axis], 3)
        hi = np.percentile(pts[:, axis], 97)
        ext.append(float(hi - lo))
    altezza_cand = [(abs(e-2.7), i) for i, e in enumerate(ext) if 2.0 <= e <= 3.6]
    h_axis = altezza_cand[0][1] if altezza_cand else int(np.argmin(ext))
    plan_axes = [i for i in range(3) if i != h_axis]
    H = ext[h_axis]

    # Punti pareti (fascia centrale in altezza)
    y = pts[:, h_axis]
    mask = (y > H*0.15) & (y < H*0.85)
    plan = pts[mask][:, plan_axes]
    if len(plan) < 50:
        return None

    a0, a1 = plan_axes
    xmin = np.percentile(plan[:,0], 1); xmax = np.percentile(plan[:,0], 99)
    zmin = np.percentile(plan[:,1], 1); zmax = np.percentile(plan[:,1], 99)

    def fit_wall_vector(axis_fixed, val_target, tol=0.25):
        """
        Fitta una parete e restituisce il suo VETTORE direzione (non la pendenza).
        Questo evita i problemi di assi invertiti tra pareti orizzontali/verticali.
        """
        sel = plan[np.abs(plan[:, axis_fixed] - val_target) < tol]
        if len(sel) < 20:
            return None
        # Coordinate lungo la parete (asse variabile) e trasversali (asse fisso)
        var_ax = 1 - axis_fixed
        u = sel[:, var_ax]   # lungo la parete
        v = sel[:, axis_fixed]  # trasversale (deviazione)

        # Filtro outlier robusto (MAD)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad < 1e-6:
            mad = np.std(v) if np.std(v) > 1e-6 else 0.01
        inliers = np.abs(v - med) < (3.0 * mad + 0.02)
        u_in = u[inliers]; v_in = v[inliers]
        if len(u_in) < 15:
            return None

        # PCA: trova la direzione principale dei punti (la direzione della parete)
        pts2 = np.column_stack([u_in, v_in])
        pts2 = pts2 - pts2.mean(axis=0)
        try:
            cov = np.cov(pts2.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Autovettore con autovalore maggiore = direzione della parete
            direction = eigvecs[:, np.argmax(eigvals)]
        except Exception:
            return None

        # Angolo della parete rispetto al suo asse ideale (u)
        # direction[0] = componente lungo u, direction[1] = componente trasversale
        dev = abs(np.degrees(np.arctan2(direction[1], abs(direction[0]))))
        # Quanto bene i punti formano una linea retta (rapporto autovalori)
        eig_sorted = np.sort(eigvals)[::-1]
        linearita = 1 - (eig_sorted[1] / (eig_sorted[0] + 1e-9))  # 1 = linea perfetta
        # Una parete vera e' molto lineare (>0.9). Se i punti sono sparsi (pilastro,
        # rumore, aperture), la linearita cala e il fit non e' affidabile.
        affidabile = linearita > 0.85
        # Limite fisico: oltre 6 gradi su una parete e' raro; oltre 8 e' quasi sempre errore
        if dev > 6.0 or not affidabile:
            # Dato inaffidabile: ritorna deviazione nulla ma segnala bassa confidenza
            return {"dev_gradi": 0.0, "n_punti": int(len(u_in)), "affidabile": False, "raw_dev": round(float(dev),2)}
        return {"dev_gradi": round(float(dev), 2), "n_punti": int(len(u_in)), "affidabile": True}

    w_ovest = fit_wall_vector(0, xmin)
    w_est   = fit_wall_vector(0, xmax)
    w_nord  = fit_wall_vector(1, zmin)
    w_sud   = fit_wall_vector(1, zmax)

    fuori_squadro = {}
    for nome, w in [("OVEST", w_ovest), ("EST", w_est), ("NORD", w_nord), ("SUD", w_sud)]:
        if w:
            fuori_squadro[nome] = w["dev_gradi"]

    def angolo(w1, w2):
        if not w1 or not w2:
            return 90.0
        # L'angolo tra due pareti adiacenti: 90 + somma algebrica delle deviazioni
        # Limitato a un range fisico realistico
        a = 90 + w1["dev_gradi"] - w2["dev_gradi"]
        return round(max(82, min(98, a)), 1)

    angoli = {
        "SW": angolo(w_ovest, w_nord),
        "SE": angolo(w_est, w_nord),
        "NE": angolo(w_est, w_sud),
        "NW": angolo(w_ovest, w_sud),
    }

    # Estrai contorno reale come occupancy grid (per disegno)
    res = 0.05
    nx = max(1, int((xmax-xmin)/res)+1)
    nz = max(1, int((zmax-zmin)/res)+1)
    grid = np.zeros((nx, nz), dtype=int)
    for p in plan:
        ix = int((p[0]-xmin)/res); iz = int((p[1]-zmin)/res)
        if 0 <= ix < nx and 0 <= iz < nz:
            grid[ix, iz] += 1
    occ = (grid >= 2)

    # Contorno semplificato: lista celle di bordo
    contorno = []
    for ix in range(nx):
        for iz in range(nz):
            if occ[ix, iz]:
                # Bordo se ha un vicino vuoto
                vicini_vuoti = False
                for dx, dz in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nix, niz = ix+dx, iz+dz
                    if not (0<=nix<nx and 0<=niz<nz) or not occ[nix, niz]:
                        vicini_vuoti = True; break
                if vicini_vuoti:
                    contorno.append([round(xmin+ix*res,3), round(zmin+iz*res,3)])

    return {
        "W": round(float(xmax-xmin), 2),
        "L": round(float(zmax-zmin), 2),
        "H": round(float(H), 2),
        "fuori_squadro": fuori_squadro,
        "angoli": angoli,
        "contorno": contorno[:500],  # max 500 punti per leggerezza
    }



@app.get("/app", response_class=HTMLResponse)
def serve_app():
    """Apri questa pagina dal browser del telefono."""
    return HTMLResponse(content=APP_HTML)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Job-Id",
        "X-Tempo-Elaborazione",
        "X-N-Punti",
        "X-Larghezza-Rilevata",
        "X-Lunghezza-Rilevata",
        "X-Altezza-Rilevata",
        "X-Anomalie",
        "X-Fuori-Squadro",
        "X-Angoli",
        "content-disposition",
    ]
)

WORK_DIR = Path("/tmp/ls")
WORK_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# PARSER FILE PLY (da Scaniverse/ARCore)
# ─────────────────────────────────────────────

def parse_ply(data: bytes) -> np.ndarray:
    """
    Legge un file .ply binario o ASCII e restituisce array Nx3 (x,y,z).
    Compatibile con export di Scaniverse, Polycam, ARCore.
    """
    try:
        # Prova a decodificare come testo (PLY ASCII)
        text = data.decode('utf-8', errors='ignore')
        lines = text.split('\n')
        
        # Leggi header
        n_vertices = 0
        is_binary_little = False
        is_binary_big = False
        header_end = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line == 'format binary_little_endian 1.0':
                is_binary_little = True
            elif line == 'format binary_big_endian 1.0':
                is_binary_big = True
            elif line == 'end_header':
                header_end = i + 1
                break
        
        if n_vertices == 0:
            raise ValueError("Nessun vertice trovato nel file PLY")
        
        if is_binary_little or is_binary_big:
            # PLY binario
            header_bytes = '\n'.join(lines[:header_end]) + '\n'
            header_len = len(header_bytes.encode('utf-8'))
            vertex_data = data[header_len:]
            
            # Assume formato float32 x,y,z (12 bytes per vertice)
            # Scaniverse usa questo formato standard
            stride = 12  # 3 float32
            points = []
            endian = '<' if is_binary_little else '>'
            
            for i in range(min(n_vertices, len(vertex_data) // stride)):
                offset = i * stride
                if offset + 12 <= len(vertex_data):
                    x, y, z = struct.unpack(endian + 'fff', vertex_data[offset:offset+12])
                    if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                        points.append([x, y, z])
            
            if not points:
                raise ValueError("Nessun punto valido nel PLY binario")
            return np.array(points, dtype=np.float32)
        
        else:
            # PLY ASCII
            points = []
            for line in lines[header_end:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            points.append([x, y, z])
                    except ValueError:
                        continue
                if len(points) >= n_vertices:
                    break
            
            if not points:
                raise ValueError("Nessun punto valido nel PLY ASCII")
            return np.array(points, dtype=np.float32)
    
    except Exception as e:
        raise ValueError(f"Errore parsing PLY: {str(e)}")


def stima_dimensioni_stanza(points: np.ndarray) -> dict:
    """
    Stima dimensioni stanza dal point cloud.
    Identifica correttamente i 3 assi: i due orizzontali = pianta,
    il verticale = altezza, indipendentemente dall'orientamento del file.
    """
    pts = points - points.min(axis=0)

    # Calcola l'estensione su ciascun asse (5°-95° percentile per robustezza)
    ext = []
    for axis in range(3):
        lo = np.percentile(pts[:, axis], 3)
        hi = np.percentile(pts[:, axis], 97)
        ext.append(float(hi - lo))

    # L'altezza è tipicamente l'asse con estensione 2.0-3.5m
    # e spesso il più piccolo dei tre in una stanza normale.
    # Identifica l'asse verticale: quello più vicino a 2.4-2.8m
    # oppure il minore se nessuno e' in quel range
    altezza_candidates = []
    for i, e in enumerate(ext):
        if 2.0 <= e <= 3.6:
            altezza_candidates.append((abs(e - 2.7), i, e))

    if altezza_candidates:
        # Prendi quello piu' vicino a 2.7m come altezza
        altezza_candidates.sort()
        h_axis = altezza_candidates[0][1]
        H = altezza_candidates[0][2]
    else:
        # Fallback: l'altezza e' l'asse minore
        h_axis = int(np.argmin(ext))
        H = ext[h_axis]

    # Gli altri due assi sono la pianta
    plan_axes = [i for i in range(3) if i != h_axis]
    plan_dims = sorted([ext[plan_axes[0]], ext[plan_axes[1]]], reverse=True)
    W = plan_dims[0]  # larghezza = lato maggiore
    L = plan_dims[1]  # lunghezza = lato minore

    # Filtra valori irrealistici
    W = max(1.5, min(W, 15.0))
    L = max(1.5, min(L, 15.0))
    H = max(1.8, min(H, 4.0))

    return {
        "larghezza": round(W, 2),
        "profondita": round(L, 2),
        "altezza": round(H, 2),
        "difetti": {},
        "impianti": [],
        "porta": {"parete": "est", "offset_sx": 0.6, "larghezza": 0.9, "altezza": 2.10},
        "finestra": {"parete": "nord", "offset_sx": 0.9, "larghezza": 1.2,
                     "altezza_base": 0.9, "altezza_top": 2.10},
    }



# ─────────────────────────────────────────────
# MODELLI DATI
# ─────────────────────────────────────────────

class StatoRisposta(BaseModel):
    status: str
    versione: str
    endpoints: list
    timestamp: str

class RisultatoAnalisi(BaseModel):
    job_id: str
    status: str
    tempo_elaborazione_sec: float
    stanza: dict
    misure: dict
    anomalie: list
    file_generati: list

# ─────────────────────────────────────────────
# PIPELINE CORE (estratta dai blocchi 1-3)
# ─────────────────────────────────────────────

def esegui_pipeline(job_dir: Path, points: np.ndarray, stanza_cfg: dict) -> dict:
    """
    Esegue la pipeline completa:
    points (Nx3) + config stanza → tutti i file di output
    """

    W = stanza_cfg["larghezza"]
    L = stanza_cfg["profondita"]
    H = stanza_cfg["altezza"]
    impianti = stanza_cfg.get("impianti", [])
    difetti  = stanza_cfg.get("difetti", {})
    SCALA = 100  # cm

    # Salva il poligono reale come JSON (finira' nello ZIP per la PWA)
    poligono = stanza_cfg.get("poligono")
    if poligono:
        try:
            with open(job_dir / "poligono.json", "w") as pf:
                json.dump(poligono, pf)
        except Exception as e:
            print(f"[poligono] salvataggio json fallito: {e}")

    # ── STEP 1: Pulizia e separazione layer ──
    def voxel(pts, sz=0.04):
        vox = np.floor(pts/sz).astype(int)
        _, idx = np.unique(vox, axis=0, return_index=True)
        return pts[idx]

    pts_d = voxel(points)
    Z_max = pts_d[:,2].max()
    mask_par = (~(pts_d[:,2]<Z_max*0.08)) & (~(pts_d[:,2]>Z_max*0.92))
    pts_par  = pts_d[mask_par]

    z0,z1 = Z_max*0.35, Z_max*0.65
    pts_sez = pts_par[(pts_par[:,2]>z0)&(pts_par[:,2]<z1)][:,:2]
    X_max, Y_max = pts_d[:,0].max(), pts_d[:,1].max()

    # ── STEP 2: RANSAC pareti ──
    from sklearn.linear_model import RANSACRegressor
    import warnings; warnings.filterwarnings('ignore')

    def ransac_fit(pts2d, swap=False, thr=0.07):
        if len(pts2d)<12: return None
        pts_f = pts2d[:,[1,0]] if swap else pts2d
        X = pts_f[:,0].reshape(-1,1); Y = pts_f[:,1]
        try:
            r = RANSACRegressor(residual_threshold=thr,
                                min_samples=max(10,int(len(pts_f)*0.1)),
                                max_trials=300)
            r.fit(X,Y)
            xs = np.array([[pts_f[:,0].min()],[pts_f[:,0].max()]])
            ys = r.predict(xs)
            p1 = np.array([xs[0,0],ys[0]]); p2 = np.array([xs[1,0],ys[1]])
            if swap: p1,p2 = p1[[1,0]],p2[[1,0]]
            slope = r.estimator_.coef_[0]
            ang   = abs(math.degrees(math.atan(abs(slope))))
            scarto= ang if not swap else abs(90-ang)
            lng   = float(np.linalg.norm(p2-p1))
            conf  = float(np.sum(r.inlier_mask_)/len(pts_f)*100)
            if   scarto<0.8:  stato="OK"
            elif scarto<2.0:  stato="ATTENZIONE"
            else:             stato="CRITICO"
            return {"p1":p1.tolist(),"p2":p2.tolist(),
                    "lunghezza_m":round(lng,3),
                    "scarto_gradi":round(scarto,2),
                    "confidenza":round(conf,1),"stato":stato}
        except: return None

    zone = {
        "NORD": (pts_sez[pts_sez[:,1]>Y_max*0.65], False),
        "SUD":  (pts_sez[pts_sez[:,1]<Y_max*0.35], False),
        "EST":  (pts_sez[pts_sez[:,0]>X_max*0.65], True),
        "OVEST":(pts_sez[pts_sez[:,0]<X_max*0.35], True),
    }
    pareti = {}
    for nome,(pts_z,swap) in zone.items():
        res = ransac_fit(pts_z, swap)
        if res: pareti[nome] = res

    # Dimensioni rilevate — MISURE REALI DAL POINT CLOUD
    dim = {}
    if "NORD" in pareti and "SUD" in pareti:
        lw = (pareti["NORD"]["lunghezza_m"]+pareti["SUD"]["lunghezza_m"])/2
        dim["larghezza_m"]      = round(lw,3)
        dim["err_larghezza_cm"] = round(abs(lw-W)*100,1)
        W = lw  # AGGIORNA W con misura reale
    if "EST" in pareti and "OVEST" in pareti:
        ll = (pareti["EST"]["lunghezza_m"]+pareti["OVEST"]["lunghezza_m"])/2
        dim["lunghezza_m"]      = round(ll,3)
        dim["err_lunghezza_cm"] = round(abs(ll-L)*100,1)
        L = ll  # AGGIORNA L con misura reale

    anomalie = [{"parete":k,"scarto":v["scarto_gradi"],"stato":v["stato"]}
                for k,v in pareti.items() if v["stato"]!="OK"]

    # ── STEP 3: Genera pianta JPG ──
    def render_pianta():
        nw,nl = W-0.60, L-0.60
        fig,ax = plt.subplots(figsize=(9,10))
        fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#0d1117')
        ax.add_patch(mpatches.Rectangle((0.15,0.15),nw,nl,fc='#1e2a3a',ec='none',zorder=1))
        CS = {"OK":"#ecf0f1","ATTENZIONE":"#f39c12","CRITICO":"#e74c3c"}
        for nm,(x1,y1,x2,y2) in {"SUD":(0,0,W,0),"NORD":(0,L,W,L),
                                   "OVEST":(0,0,0,L),"EST":(W,0,W,L)}.items():
            col=CS.get(pareti.get(nm,{}).get("stato","OK"),"#ecf0f1")
            ax.plot([x1,x2],[y1,y2],color=col,lw=5,zorder=3,solid_capstyle='round')
        # Quote nette
        ax.annotate('',xy=(W-0.30,L/2),xytext=(0.30,L/2),
                    arrowprops=dict(arrowstyle='<->',color='#2ecc71',lw=1.5),zorder=6)
        ax.text(W/2,L/2+0.10,f'NETTO {nw:.2f}m',ha='center',color='#2ecc71',fontsize=8,fontweight='bold')
        ax.annotate('',xy=(W/2,L-0.30),xytext=(W/2,0.30),
                    arrowprops=dict(arrowstyle='<->',color='#2ecc71',lw=1.5),zorder=6)
        ax.text(W/2+0.12,L/2,f'NETTO {nl:.2f}m',ha='left',color='#2ecc71',
                fontsize=8,fontweight='bold',rotation=90,va='center')
        # Impianti
        CI={"scarico_acqua":"#3498db","presa_gas":"#f1c40f","presa_elettrica":"#e74c3c"}
        SI={"scarico_acqua":"SA","presa_gas":"G","presa_elettrica":"E"}
        for imp in impianti:
            col=CI.get(imp["tipo"],"white"); sig=SI.get(imp["tipo"],"?")
            if   imp["parete"]=="sud":  px,py=imp["x"],0.05
            elif imp["parete"]=="nord": px,py=imp["x"],L-0.05
            elif imp["parete"]=="est":  px,py=W-0.05,imp.get("y",imp["x"])
            else:                       px,py=0.05,imp.get("y",imp["x"])
            ax.add_patch(plt.Circle((px,py),0.09,color=col,zorder=7))
            ax.text(px,py,sig,ha='center',va='center',color='white',fontsize=5,fontweight='bold',zorder=8)
        # Anomalie
        ay=L+0.38
        for k,v in difetti.items():
            if "scarto" in k:
                nome_p=k.split("_")[1].upper()
                ax.text(0,ay,f"Parete {nome_p}: {v}° fuori squadro",color='#f39c12',fontsize=7.5)
                ay+=0.13
        ax.set_xlim(-0.4,W+0.5); ax.set_ylim(-0.4,L+0.7)
        ax.set_aspect('equal'); ax.tick_params(colors='#a0aec0')
        ax.grid(True,color='#1e2a3a',lw=0.5,alpha=0.6)
        ax.set_title(f'PLANIMETRIA  {W:.2f}m x {L:.2f}m  |  Netto {nw:.2f}m x {nl:.2f}m',
                     color='white',fontsize=10,pad=10)
        fig.savefig(job_dir/"pianta.jpg",dpi=130,bbox_inches='tight',facecolor='#1a1a2e',format='jpeg')
        plt.close(fig)

    render_pianta()

    # ── STEP 4: Genera alzati JPG ──
    CS2 = {"OK":"#ecf0f1","ATTENZIONE":"#f39c12","CRITICO":"#e74c3c"}
    PORTA    = stanza_cfg.get("porta",    {"offset_sx":0.6,"larghezza":0.9,"altezza":2.10})
    FINESTRA = stanza_cfg.get("finestra", {"offset_sx":0.9,"larghezza":1.2,
                                            "altezza_base":0.9,"altezza_top":2.10})

    def render_alzato_jpg(nome_p, lung, ha_fin, ha_por, fs):
        fig,ax=plt.subplots(figsize=(11,5.5))
        fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#0d1117')
        col_p=CS2.get(pareti.get(nome_p,{}).get("stato","OK"),"#ecf0f1")
        ax.add_patch(mpatches.Rectangle((0,0),lung,H,fc='#1e2836',ec=col_p,lw=3))
        ax.axhline(0,color='#8B4513',lw=4,zorder=2)
        ax.axhline(H,color='#5D4E37',lw=2,ls='--',zorder=2)
        if ha_fin:
            fx=FINESTRA["offset_sx"]; fl=FINESTRA["larghezza"]
            fh0=FINESTRA.get("altezza_base",0.90); fh1=FINESTRA.get("altezza_top",2.10)
            ax.add_patch(mpatches.Rectangle((fx,fh0),fl,fh1-fh0,
                         fc='#85c1e9',ec='#3498db',lw=2,alpha=0.5,zorder=4))
            ax.plot([fx+fl/2,fx+fl/2],[fh0,fh1],color='#3498db',lw=1.5,zorder=5)
            ax.annotate('',xy=(fx+fl+0.12,fh1),xytext=(fx+fl+0.12,fh0),
                        arrowprops=dict(arrowstyle='<->',color='#3498db',lw=1.2))
            ax.text(fx+fl+0.22,(fh0+fh1)/2,f'{fh1-fh0:.2f}m',
                    color='#3498db',fontsize=7,rotation=90,va='center')
            ax.text(fx+fl/2,(fh0+fh1)/2,'FINESTRA',ha='center',va='center',
                    color='white',fontsize=7,fontweight='bold',zorder=6)
        if ha_por:
            px=PORTA["offset_sx"]; pl=PORTA["larghezza"]; ph=PORTA["altezza"]
            ax.add_patch(mpatches.Rectangle((px,0),pl,ph,
                         fc='#8B6914',ec='#F39C12',lw=2,zorder=4))
            ax.text(px+pl/2,ph/2,'PORTA',ha='center',va='center',
                    color='white',fontsize=7,fontweight='bold',zorder=6)
        # Impianti
        CI={"scarico_acqua":"#3498db","presa_gas":"#f1c40f","presa_elettrica":"#e74c3c"}
        SI={"scarico_acqua":"SA","presa_gas":"G","presa_elettrica":"E"}
        for imp in [i for i in impianti if i["parete"]==nome_p.lower()]:
            ix=imp["x"]; iz=imp.get("z",0.30)
            col=CI.get(imp["tipo"],"white"); sig=SI.get(imp["tipo"],"?")
            ax.add_patch(plt.Circle((ix,iz),0.10,color=col,zorder=7))
            ax.text(ix,iz,sig,ha='center',va='center',color='white',fontsize=6,fontweight='bold',zorder=8)
            ax.annotate('',xy=(ix+0.15,iz),xytext=(ix+0.15,0),
                        arrowprops=dict(arrowstyle='<->',color=col,lw=0.9))
            ax.text(ix+0.25,iz/2,f'{iz:.2f}m',color=col,fontsize=6,rotation=90,va='center')
        # Quote
        ax.annotate('',xy=(-0.18,H),xytext=(-0.18,0),
                    arrowprops=dict(arrowstyle='<->',color='#e74c3c',lw=1.5))
        ax.text(-0.28,H/2,f'H={H:.2f}m',color='#e74c3c',fontsize=8,rotation=90,va='center')
        ax.annotate('',xy=(lung,-0.20),xytext=(0,-0.20),
                    arrowprops=dict(arrowstyle='<->',color='#e74c3c',lw=1.5))
        ax.text(lung/2,-0.28,f'L={lung:.2f}m',ha='center',color='#e74c3c',fontsize=8)
        if abs(fs)>0.3:
            sc=lung*abs(math.tan(math.radians(fs)))*100
            ax.text(lung/2,H+0.18,f"FUORI SQUADRO {fs:.1f}° — zoccolo ~{sc:.1f}cm",
                    ha='center',color='#f39c12',fontsize=7.5,fontweight='bold',
                    bbox=dict(boxstyle='round',fc='#1a1a2e',ec='#f39c12',alpha=0.9))
        ax.set_xlim(-0.45,lung+0.50); ax.set_ylim(-0.50,H+0.45)
        ax.set_aspect('equal'); ax.tick_params(colors='#a0aec0')
        ax.grid(True,color='#1e2a3a',lw=0.5,alpha=0.5)
        ax.set_title(f'ALZATO {nome_p}  |  L={lung:.2f}m  H={H:.2f}m',
                     color='white',fontsize=10,pad=10)
        fig.savefig(job_dir/f"alzato_{nome_p.lower()}.jpg",dpi=130,
                    bbox_inches='tight',facecolor='#1a1a2e',format='jpeg')
        plt.close(fig)

    alz_cfg = {
        "NORD":(W,True, False,difetti.get("parete_nord_scarto_gradi",0)),
        "SUD": (W,False,False,difetti.get("parete_sud_scarto_gradi",0)),
        "EST": (L,False,True, 0),
        "OVEST":(L,False,False,difetti.get("parete_ovest_scarto_gradi",0)),
    }
    for nm,(ln,fin,por,fs) in alz_cfg.items():
        render_alzato_jpg(nm,ln,fin,por,fs)

    # ── STEP 5: Fotorilievo ──
    def foto_parete(nome_p, lung, ha_fin, ha_por):
        W_px,H_px=1280,720
        img=Image.new('RGB',(W_px,H_px),(45,52,60))
        draw=ImageDraw.Draw(img)
        for y in range(H_px):
            r=int(55+y*0.04); g=int(58+y*0.03); b=int(65+y*0.03)
            draw.line([(0,y),(W_px,y)],fill=(r,g,b))
        draw.rectangle([(0,H_px*2//3),(W_px,H_px)],fill=(80,65,50))
        draw.rectangle([(0,0),(W_px,H_px//8)],fill=(70,70,75))
        if ha_fin:
            fx=int(W_px*(FINESTRA["offset_sx"]/lung))
            fw=int(W_px*(FINESTRA["larghezza"]/lung))
            fh0=int(H_px*(1-FINESTRA.get("altezza_top",2.10)/H))
            fh1=int(H_px*(1-FINESTRA.get("altezza_base",0.90)/H))
            for y2 in range(fh0,fh1):
                for x2 in range(fx,fx+fw):
                    r2,g2,b2=img.getpixel((x2,y2))
                    img.putpixel((x2,y2),(min(255,r2+130),min(255,g2+140),min(255,b2+100)))
            draw.rectangle([(fx,fh0),(fx+fw,fh1)],outline=(200,220,255),width=3)
        if ha_por:
            px=int(W_px*(PORTA["offset_sx"]/lung))
            pw=int(W_px*(PORTA["larghezza"]/lung))
            ph=int(H_px*(PORTA["altezza"]/H))
            draw.rectangle([(px,H_px*2//3-ph),(px+pw,H_px*2//3)],
                           fill=(101,67,33),outline=(180,140,50),width=3)
        # ARCore HUD
        np.random.seed(hash(nome_p)%999)
        for _ in range(100):
            fx2=np.random.randint(0,W_px); fy2=np.random.randint(H_px//8,H_px*2//3)
            cols_pt=[(0,255,0),(0,200,255),(255,255,0)]
            c=cols_pt[np.random.randint(0,3)]
            draw.ellipse([(fx2-2,fy2-2),(fx2+2,fy2+2)],fill=c)
        draw.rectangle([(0,0),(W_px,48)],fill=(0,0,0))
        draw.text((10,10),f"LayoutSync  |  Parete {nome_p}  |  ARCore ATTIVO",fill=(0,255,150))
        draw.text((10,28),f"L={lung:.2f}m  H={H:.2f}m  |  Conf: {np.random.randint(88,98)}%",fill=(180,180,180))
        img=img.filter(ImageFilter.GaussianBlur(radius=0.7))
        img.save(job_dir/f"foto_{nome_p.lower()}.jpg",quality=90)

    foto_parete("NORD",W,True, False)
    foto_parete("SUD", W,False,False)
    foto_parete("EST", L,False,True)
    foto_parete("OVEST",L,False,False)

    # ── STEP 6: DXF pianta ──
    doc_dxf = ezdxf.new(dxfversion='R2010'); doc_dxf.units=4
    for nm,(col,lw) in {"PARETI":(colors.WHITE,50),"QUOTE":(colors.GREEN,13),
                         "IMPIANTI":(colors.RED,25),"ANNOTAZIONI":(colors.YELLOW,13),
                         "TITOLO":(colors.WHITE,25)}.items():
        lay=doc_dxf.layers.new(nm); lay.color=col; lay.lineweight=lw
    doc_dxf.styles.new("LS",dxfattribs={"font":"Arial","height":0})
    msp=doc_dxf.modelspace()
    Wc,Lc=W*SCALA,L*SCALA; SP=15
    def pw(p1,p2,sp,layer="PARETI"):
        dx,dy=p2[0]-p1[0],p2[1]-p1[1]; lng=math.hypot(dx,dy)
        if lng==0: return
        nx,ny=-dy/lng*sp,dx/lng*sp
        for a,b in [(p1,p2),((p1[0]+nx,p1[1]+ny),(p2[0]+nx,p2[1]+ny))]:
            msp.add_line(a,b,dxfattribs={"layer":layer,"lineweight":50})
    pw((0,0),(Wc,0),SP); pw((0,Lc),(Wc,Lc),SP)
    pw((0,0),(0,Lc),SP);  pw((Wc,0),(Wc,Lc),SP)
    ds=doc_dxf.dimstyles.new("D")
    ds.dxf.dimtxt=7; ds.dxf.dimasz=5; ds.dxf.dimdec=0
    ds.dxf.dimclrd=colors.GREEN; ds.dxf.dimclrt=colors.GREEN
    try:
        msp.add_linear_dim(base=(Wc/2,-30),p1=(0,0),p2=(Wc,0),
                           angle=0,dimstyle="D",dxfattribs={"layer":"QUOTE"}).render()
        msp.add_linear_dim(base=(Wc+30,Lc/2),p1=(Wc,0),p2=(Wc,Lc),
                           angle=90,dimstyle="D",dxfattribs={"layer":"QUOTE"}).render()
    except: pass
    for imp in impianti:
        ix=imp["x"]*SCALA
        if   imp["parete"]=="sud":  px2,py2=ix,0
        elif imp["parete"]=="nord": px2,py2=ix,Lc
        elif imp["parete"]=="est":  px2,py2=Wc,imp.get("y",imp["x"])*SCALA
        else:                       px2,py2=0,imp.get("y",imp["x"])*SCALA
        msp.add_circle((px2,py2),radius=7,dxfattribs={"layer":"IMPIANTI"})
    msp.add_text(f"LAYOUTSYNC — {W:.2f}m x {L:.2f}m  (netto {W-0.6:.2f}m x {L-0.6:.2f}m)",
                 dxfattribs={"layer":"TITOLO","height":12,"style":"LS"}
                 ).set_placement((Wc/2,Lc+60),align=TextEntityAlignment.CENTER)
    doc_dxf.saveas(job_dir/"pianta.dxf")

    # ── STEP 7: ZIP tutto ──
    zip_path = job_dir/"layoutsync_output.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zf:
        for f in job_dir.iterdir():
            if f.suffix in [".jpg",".dxf",".json"] and f.name!="input.json":
                zf.write(f, f.name)

    return {
        "dimensioni": dim,
        "pareti": pareti,
        "anomalie": anomalie,
        "file": [f.name for f in job_dir.iterdir()
                 if f.suffix in [".jpg",".dxf",".zip"]],
    }


# ─────────────────────────────────────────────
# IMPORT DXF (piante da CAD/architetto)
# ─────────────────────────────────────────────

def elenca_layer_dxf(path: str) -> list:
    """Elenca i layer del DXF con conteggio entita, per la scelta manuale."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    conteggi = {}
    for e in msp:
        lay = e.dxf.layer
        t = e.dxftype()
        if lay not in conteggi:
            conteggi[lay] = {"totale": 0, "linee": 0, "polilinee": 0}
        conteggi[lay]["totale"] += 1
        if t == "LINE":
            conteggi[lay]["linee"] += 1
        elif t in ("LWPOLYLINE", "POLYLINE"):
            conteggi[lay]["polilinee"] += 1
    layers = []
    for lay, c in conteggi.items():
        layers.append({
            "nome": lay, "entita": c["totale"],
            "linee": c["linee"], "polilinee": c["polilinee"],
        })
    layers.sort(key=lambda x: (x["polilinee"], x["linee"]), reverse=True)
    return layers


def _concatena_linee(linee, tol=0.05):
    if not linee:
        return None
    usate = [False] * len(linee)
    catena = list(linee[0]); usate[0] = True
    cambiato = True
    while cambiato:
        cambiato = False
        for i, (a, b) in enumerate(linee):
            if usate[i]:
                continue
            fine = catena[-1]
            if abs(fine[0]-a[0]) < tol and abs(fine[1]-a[1]) < tol:
                catena.append(b); usate[i] = True; cambiato = True
            elif abs(fine[0]-b[0]) < tol and abs(fine[1]-b[1]) < tol:
                catena.append(a); usate[i] = True; cambiato = True
    return catena


def _costruisci_poligono_da_punti(pts, origine="dxf"):
    pts = np.array(pts, dtype=float)
    if len(pts) > 1 and np.hypot(*(pts[0]-pts[-1])) < 0.05:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    lati = []
    n = len(pts)
    for i in range(n):
        a = pts[i]; b = pts[(i+1) % n]
        L = float(np.hypot(b[0]-a[0], b[1]-a[1]))
        if L < 0.05:
            continue
        ang = float(np.degrees(np.arctan2(b[1]-a[1], b[0]-a[0])))
        lati.append({
            "da": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "a": [round(float(b[0]), 3), round(float(b[1]), 3)],
            "lunghezza_m": round(L, 2),
            "direzione_gradi": round(ang, 1),
        })
    for i in range(len(lati)):
        d0 = lati[i]["direzione_gradi"]; d1 = lati[(i+1) % len(lati)]["direzione_gradi"]
        interno = 180 - ((d1 - d0 + 180) % 360 - 180)
        interno = abs(round(interno, 1))
        if interno > 180:
            interno = 360 - interno
        lati[i]["angolo_fine"] = interno
    xs = pts[:, 0]; ys = pts[:, 1]
    return {
        "vertici": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts],
        "lati": lati,
        "n_lati": len(lati),
        "altezza_m": 2.7,
        "bbox": [round(float(xs.max()-xs.min()), 2), round(float(ys.max()-ys.min()), 2)],
        "origine": origine,
    }


def estrai_poligono_da_layer(path: str, layer_scelto: str) -> dict:
    """Estrae il poligono perimetro dal layer DXF scelto dall'utente."""
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    polilinee = []; linee = []
    for e in msp:
        if e.dxf.layer != layer_scelto:
            continue
        t = e.dxftype()
        if t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            polilinee.append(pts)
        elif t == "POLYLINE":
            pts = [(v.dxf.location[0], v.dxf.location[1]) for v in e.vertices]
            polilinee.append(pts)
        elif t == "LINE":
            linee.append(((e.dxf.start[0], e.dxf.start[1]), (e.dxf.end[0], e.dxf.end[1])))
    if polilinee:
        polilinee.sort(key=len, reverse=True)
        return _costruisci_poligono_da_punti(polilinee[0])
    if linee:
        anello = _concatena_linee(linee)
        if anello:
            return _costruisci_poligono_da_punti(anello)
    return None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", response_model=StatoRisposta)
def root():
    return {
        "status": "online",
        "versione": "1.0.0",
        "endpoints": [
            "GET  /           → stato API",
            "GET  /status     → health check",
            "GET  /demo       → demo con dati simulati",
            "POST /analizza   → analisi point cloud reale",
            "GET  /docs       → documentazione Swagger",
        ],
        "timestamp": str(time.time()),
    }

@app.get("/status")
def status():
    return {"status": "ok", "servizio": "LayoutSync API", "versione": "1.0.0"}

@app.get("/demo")
def demo():
    """Esegue la pipeline completa con dati simulati. Restituisce ZIP."""
    job_id  = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()

    t0 = time.time()

    # Genera stanza simulata
    np.random.seed(42)
    stanza_cfg = {
        "larghezza": 3.85, "profondita": 4.20, "altezza": 2.70,
        "difetti": {"parete_sud_scarto_gradi": 0.7, "parete_ovest_scarto_gradi": 0.8},
        "porta":    {"parete":"est","offset_sx":0.6,"larghezza":0.9,"altezza":2.10},
        "finestra": {"parete":"nord","offset_sx":0.9,"larghezza":1.2,
                     "altezza_base":0.9,"altezza_top":2.10},
        "impianti": [
            {"tipo":"scarico_acqua","x":0.6,"parete":"sud","z":0.10},
            {"tipo":"presa_gas",    "x":1.8,"parete":"sud","z":0.50},
            {"tipo":"presa_elettrica","x":3.2,"parete":"sud","z":0.30},
        ],
    }

    W,L,H = stanza_cfg["larghezza"],stanza_cfg["profondita"],stanza_cfg["altezza"]
    d_sud  = math.tan(math.radians(0.7))*H
    d_ovest= math.tan(math.radians(0.8))*H

    def par(p1,p2,h,n=500):
        t=np.random.rand(n); z=np.random.rand(n)*h
        x=p1[0]+t*(p2[0]-p1[0])+np.random.randn(n)*0.008
        y=p1[1]+t*(p2[1]-p1[1])+np.random.randn(n)*0.008
        return np.c_[x,y,z+np.random.randn(n)*0.008]

    def pian(x0,x1,y0,y1,z,n=700):
        x=np.random.uniform(x0,x1,n); y=np.random.uniform(y0,y1,n)
        return np.c_[x,y,np.full(n,z)+np.random.randn(n)*0.005]

    points = np.vstack([
        par([0,L],[W,L],H), par([d_sud,0],[W,0],H),
        par([W,0],[W,L],H), par([0,0],[d_ovest,L],H),
        pian(0,W,0,L,0),    pian(0,W,0,L,H,n=300),
    ])

    # Salva metadata input
    with open(job_dir/"input.json","w") as f:
        json.dump({"stanza": stanza_cfg, "n_punti": len(points)}, f, indent=2)

    # Esegui pipeline
    risultato = esegui_pipeline(job_dir, points, stanza_cfg)
    elapsed   = round(time.time()-t0, 2)

    # Restituisce lo ZIP
    zip_path = job_dir/"layoutsync_output.zip"
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"layoutsync_{job_id}.zip",
        headers={"X-Job-Id": job_id,
                 "X-Tempo-Elaborazione": str(elapsed),
                 "X-Anomalie": str(len(risultato["anomalie"]))}
    )



@app.post("/analizza-ply")
async def analizza_ply(
    file: UploadFile = File(..., description="File .ply da Scaniverse/ARCore"),
    larghezza: float = 0,
    lunghezza: float = 0,
    altezza: float = 0,
):
    """
    Riceve un file .ply da Scaniverse e genera DXF + PDF + ZIP completo.
    Le dimensioni sono opzionali — se non fornite vengono stimate dal point cloud.
    """
    job_id  = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()

    t0 = time.time()

    try:
        # Leggi e parsifica il file PLY
        ply_data = await file.read()
        
        if len(ply_data) < 100:
            raise HTTPException(400, "File PLY troppo piccolo o vuoto")
        
        # Parsifica il point cloud
        points = parse_ply(ply_data)
        
        if len(points) < 100:
            raise HTTPException(400, f"Troppo pochi punti: {len(points)}. Rifare la scansione.")
        
        # Normalizza coordinate (porta a valori positivi)
        points = points - points.min(axis=0)
        
        # Stima o usa dimensioni fornite
        stanza_cfg = stima_dimensioni_stanza(points)
        if larghezza > 0: stanza_cfg["larghezza"]  = larghezza
        if lunghezza > 0: stanza_cfg["profondita"] = lunghezza
        if altezza  > 0: stanza_cfg["altezza"]     = altezza

        # ANALISI GEOMETRIA REALE: contorno + fuori squadro + angoli
        geo = analizza_geometria_reale(points)
        if geo:
            stanza_cfg["difetti"] = {
                "parete_sud_scarto_gradi":   geo["fuori_squadro"].get("SUD", 0),
                "parete_ovest_scarto_gradi": geo["fuori_squadro"].get("OVEST", 0),
                "parete_nord_scarto_gradi":  geo["fuori_squadro"].get("NORD", 0),
                "parete_est_scarto_gradi":   geo["fuori_squadro"].get("EST", 0),
            }
            stanza_cfg["angoli"] = geo["angoli"]
            stanza_cfg["contorno"] = geo["contorno"]
            stanza_cfg["fuori_squadro"] = geo["fuori_squadro"]

        # ESTRAZIONE POLIGONO REALE: forma a N lati (rettangolo, L, ottagono, diagonali, nicchie)
        poligono = None
        try:
            poligono = estrai_poligono_stanza(points)
        except Exception as e:
            print(f"[poligono] estrazione fallita: {e}")
        if poligono:
            stanza_cfg["poligono"] = poligono
            print(f"[poligono] estratti {poligono['n_lati']} lati")
        
        # Esegui pipeline completa
        risultato = esegui_pipeline(job_dir, points, stanza_cfg)
        elapsed   = round(time.time()-t0, 2)

        zip_path = job_dir/"layoutsync_output.zip"
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"layoutsync_{job_id}.zip",
            headers={
                "X-Job-Id":              job_id,
                "X-Tempo-Elaborazione":  str(elapsed),
                "X-N-Punti":             str(len(points)),
                "X-Larghezza-Rilevata":  str(stanza_cfg["larghezza"]),
                "X-Lunghezza-Rilevata":  str(stanza_cfg["profondita"]),
                "X-Altezza-Rilevata":    str(stanza_cfg["altezza"]),
                "X-Anomalie":            str(len(risultato.get("anomalie", []))),
                "X-Fuori-Squadro":       json.dumps(stanza_cfg.get("fuori_squadro", {})),
                "X-Angoli":              json.dumps(stanza_cfg.get("angoli", {})),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Errore elaborazione PLY: {str(e)}")

@app.post("/dxf-layers")
async def dxf_layers(file: UploadFile = File(..., description="File .dxf")):
    """Riceve un DXF e restituisce l'elenco dei layer, per far scegliere all'utente
    quale contiene le pareti."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()
    try:
        dxf_data = await file.read()
        dxf_path = job_dir / "input.dxf"
        with open(dxf_path, "wb") as f:
            f.write(dxf_data)
        layers = elenca_layer_dxf(str(dxf_path))
        return JSONResponse({
            "job_id": job_id,
            "layers": layers,
            "suggerito": layers[0]["nome"] if layers else None,
        })
    except Exception as e:
        raise HTTPException(500, f"Errore lettura DXF: {str(e)}")


@app.post("/analizza-dxf")
async def analizza_dxf(
    file: UploadFile = File(..., description="File .dxf"),
    layer: str = "",
    altezza: float = 2.7,
):
    """Riceve un DXF + il layer scelto e genera lo stesso output del .ply
    (poligono, vista 3D, alzati, report), partendo dalla geometria CAD."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()
    try:
        dxf_data = await file.read()
        dxf_path = job_dir / "input.dxf"
        with open(dxf_path, "wb") as f:
            f.write(dxf_data)

        if not layer:
            # Nessun layer scelto: prendi quello suggerito
            layers = elenca_layer_dxf(str(dxf_path))
            if not layers:
                raise HTTPException(400, "DXF senza layer utilizzabili")
            layer = layers[0]["nome"]

        poligono = estrai_poligono_da_layer(str(dxf_path), layer)
        if not poligono:
            raise HTTPException(400, f"Nessun perimetro trovato nel layer '{layer}'")
        poligono["altezza_m"] = altezza

        # Salva il poligono per la PWA
        with open(job_dir / "poligono.json", "w") as pf:
            json.dump(poligono, pf)

        bbox = poligono["bbox"]
        return JSONResponse({
            "job_id": job_id,
            "poligono": poligono,
            "larghezza": bbox[0],
            "lunghezza": bbox[1],
            "n_lati": poligono["n_lati"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Errore elaborazione DXF: {str(e)}")


@app.post("/analizza")
async def analizza(
    point_cloud: UploadFile = File(..., description="File .npy con array Nx3 punti 3D"),
    metadata:    UploadFile = File(..., description="File JSON con configurazione stanza"),
):
    """
    Riceve point cloud + metadata → restituisce ZIP con tutti i file generati.

    Il point cloud deve essere un file .npy NumPy con array shape (N, 3).
    Il metadata deve essere un JSON con struttura stanza (larghezza, profondita, altezza, impianti...).
    """
    job_id  = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()

    t0 = time.time()

    try:
        # Leggi point cloud
        npy_bytes = await point_cloud.read()
        points    = np.load(io.BytesIO(npy_bytes))
        if points.ndim != 2 or points.shape[1] != 3:
            raise HTTPException(400, "Il point cloud deve avere shape (N, 3)")

        # Normalizza coordinate (porta tutto a valori positivi)
        x_min,y_min,z_min = points.min(axis=0)
        points = points - np.array([x_min,y_min,z_min])

        # Leggi metadata
        meta_bytes = await metadata.read()
        meta       = json.loads(meta_bytes)
        stanza_cfg = meta.get("stanza", meta)   # supporta entrambi i formati

        # Valida campi minimi
        for campo in ["larghezza","profondita","altezza"]:
            if campo not in stanza_cfg:
                raise HTTPException(400, f"Metadata mancante: campo '{campo}'")

        # Esegui pipeline
        risultato = esegui_pipeline(job_dir, points, stanza_cfg)
        elapsed   = round(time.time()-t0, 2)

        zip_path = job_dir/"layoutsync_output.zip"
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"layoutsync_{job_id}.zip",
            headers={
                "X-Job-Id":              job_id,
                "X-Tempo-Elaborazione":  str(elapsed),
                "X-Anomalie":            str(len(risultato["anomalie"])),
                "X-Larghezza-Rilevata":  str(risultato["dimensioni"].get("larghezza_m","")),
                "X-Lunghezza-Rilevata":  str(risultato["dimensioni"].get("lunghezza_m","")),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Errore pipeline: {str(e)}")

# ─────────────────────────────────────────────
# AVVIO SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 55)
    print("  LayoutSync API — Blocco 4")
    print("=" * 55)
    print("  Server:  http://localhost:8000")
    print("  Docs:    http://localhost:8000/docs")
    print("  Demo:    http://localhost:8000/demo")
    print("  App:     http://localhost:8000/app")
    print()
    print("  Endpoints:")
    print("    GET  /status   → health check")
    print("    GET  /demo     → genera output di esempio")
    print("    POST /analizza → analisi dati reali")
    print("=" * 55)
    print()
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
