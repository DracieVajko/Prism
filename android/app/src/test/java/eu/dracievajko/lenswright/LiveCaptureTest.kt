package eu.dracievajko.lenswright

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean

/**
 * JVM unit tests for the Stop-drain contract of [LiveCapture].
 *
 * A fake commit queue is injected through the test hooks, so no Android
 * framework, socket or audio device is involved: the tests exercise only
 * [LiveCapture.awaitPending] and the in-flight-commit counter it waits on.
 */
class LiveCaptureTest {

    /**
     * With one in-flight commit, awaitPending(2000) must block until the
     * `...transcription.completed` callback (simulated here by
     * completePendingForTest) arrives, and only then return true.
     */
    @Test
    fun awaitPending_waits_for_completed_callback_before_returning_true() {
        // Inject a fake commit queue holding exactly one in-flight commit.
        LiveCapture.setPendingForTest(1)
        val completed = AtomicBoolean(false)

        // Simulate the server's completed callback arriving on a background
        // thread after awaitPending has started waiting.
        val completer = Thread {
            Thread.sleep(250)
            completed.set(true)
            LiveCapture.completePendingForTest()
        }
        completer.start()

        val startedAt = System.currentTimeMillis()
        val drained = LiveCapture.awaitPending(2000)
        val elapsed = System.currentTimeMillis() - startedAt
        // Capture this before join(): awaitPending must not have returned
        // before the completed callback ran.
        val sawCompletionBeforeReturning = completed.get()
        completer.join()

        assertTrue(
            "awaitPending must return true once the in-flight commit completes",
            drained,
        )
        assertTrue(
            "awaitPending returned before the completed callback arrived",
            sawCompletionBeforeReturning,
        )
        assertTrue(
            "awaitPending should block until completion (elapsed ${elapsed}ms)",
            elapsed >= 200,
        )
    }

    /** With a commit that never completes, awaitPending(2000) must time out. */
    @Test
    fun awaitPending_returns_false_on_timeout() {
        LiveCapture.setPendingForTest(1)
        // No completed callback is ever delivered: the bounded drain must
        // give up and report false instead of hanging.
        val drained = LiveCapture.awaitPending(2000)
        assertFalse(
            "awaitPending must return false when the commit queue never drains",
            drained,
        )
        // Leave the singleton clean for any other test.
        LiveCapture.completePendingForTest()
    }
}
