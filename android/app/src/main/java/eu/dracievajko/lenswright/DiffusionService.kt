package eu.dracievajko.lenswright

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
        // Live capture is owned by this service so it survives backgrounding.
        LiveCapture.attach(applicationContext)
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
            .setContentTitle("Lenswright")
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
        // LiveCapture.stop() drains in-flight commits (bounded ~5 s) and issues
        // the session-close POST, so it must never run on the main thread (ANR
        // risk). The local API server has to stay up until that close POST has
        // been sent, so the whole teardown runs on one background thread in the
        // original order: drain/close first, then server shutdown. Thread { }.start()
        // is the same off-main pattern used in MainActivity.
        val api = server
        server = null
        Thread {
            LiveCapture.stop()
            api?.stop()
        }.start()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
