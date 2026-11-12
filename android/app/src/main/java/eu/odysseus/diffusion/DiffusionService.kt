package eu.odysseus.diffusion

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.Context
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * ForegroundService keeping the Ktor server alive.
 * The notification shows the real state (port + mode).
 */
class DiffusionService : Service() {
    private var server: ApiServer? = null

    override fun onCreate() {
        super.onCreate()
        val ch = NotificationChannel("srv", "Diffusion server", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(ch)
        try {
            startForeground(1, buildNotification("Server running on :8101"))
        } catch (e: Throwable) {
            // Android 12+ forbids promoting a service to foreground when the app
            // was started from the background (adb, widgets, automation):
            //   ForegroundServiceStartNotAllowedException
            // The server is still useful without the notification, so keep
            // running instead of taking the process down with it.
            android.util.Log.w("DiffusionService", "startForeground not allowed: ${e.message}")
        }
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, "srv")
            .setContentTitle("Prism")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (server == null) {
            server = ApiServer(AppGraph.manager, ModelDownloader(this), applicationContext).also { it.start(wait = false) }
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .notify(1, buildNotification("Server running on :8101"))
        }
        return START_STICKY
    }

    override fun onDestroy() {
        server?.stop()
        server = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
