Use this as the Unity WebGL template for the embedded bridge page.

Expected location in the real Unity project:
- `Assets/WebGLTemplates/sign_language_bridge/index.html`

After copying it into the Unity project:
1. Open `Project Settings > Player > Resolution and Presentation > WebGL Template`
2. Select `sign_language_bridge`
3. Rebuild WebGL

This template preserves:
- `unity-ready` postMessage back to the parent page
- queued bridge messages before Unity finishes loading
- forwarding of `set-source-path`
- forwarding of `set-playback-time`
- forwarding of `set-playback-state`
- forwarding of `set-playback-rate`
