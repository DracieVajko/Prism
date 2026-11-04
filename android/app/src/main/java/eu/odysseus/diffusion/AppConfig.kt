package eu.odysseus.diffusion

/**
 * Runtime configuration for the app.
 * Everything runs locally on the phone; there is no remote/PC mode.
 */
object AppConfig {
    @Volatile var serverPort: Int = 8101

    /** "auto" | "cpu" | "gpu" — selects ONNX execution providers.
     *  "gpu" uses NNAPI, which is deprecated in Android 15; prefer "auto". */
    @Volatile var acceleration: String = "auto"

    /** Optional bearer token for LAN access. Empty disables auth (default). */
    @Volatile var authToken: String = ""
}
