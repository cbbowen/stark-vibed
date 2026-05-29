@echo off
REM Serve the Stark web UI (Dioxus web + WebGPU). Requires dioxus-cli (`dx`):
REM   cargo install dioxus-cli
REM Open the printed localhost URL in a WebGPU-capable browser (recent Chrome/Edge).
dx serve --web -p stark-ui %*
