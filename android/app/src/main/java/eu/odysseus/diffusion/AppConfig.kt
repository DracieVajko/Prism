package eu.odysseus.diffusion

/**
 * Runtime configuration for the app.
 *
 * The app runs locally by default (offline, lightweight models only).
 * Optionally it can connect to a PC server (local or Tailscale) for
 * heavy features (FLUX restoration, large models, etc.).
 */
object AppConfig {
    @Volatile var serverPort: Int = 8101

    /** "auto" | "cpu" | "gpu" — selects ONNX execution providers.
     *  "gpu" uses NNAPI, which is deprecated in Android 15; prefer "auto". */
    @Volatile var acceleration: String = "auto"

    /** Optional bearer token for LAN access. Empty disables auth (default). */
    @Volatile var authToken: String = ""

    /** Connection mode: "local" (offline), "remote" (forward to PC), "auto" (try local, fall back to PC). */
    @Volatile var connectionMode: String = "local"

    /** PC server URL when in remote/auto mode (e.g. "http://192.168.1.100:8101" or "http://100.x.y.z:8101" for Tailscale). */
    @Volatile var remoteHost: String = ""
}
