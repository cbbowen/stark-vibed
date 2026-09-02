//! A session's stop signal, held by everything the session spawns — its own
//! module because everything holds one: the backend that mints it, the
//! resolvers and loops that race against it, and the [`Events`](crate::Events)
//! stream that reports the end to the engine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::Notify;

/// A session's stop signal, held by everything it spawned.
///
/// What this replaces is an inference. The unbounded retries used to key off
/// `Waitlist::is_live` — "does the UI still hold the `Events` receiver" — which is
/// a frontend convention standing in for a session fact, and left
/// [`CollabSession::shutdown`](crate::CollabSession::shutdown) cancelling nothing
/// it had spawned. Both facts are real and either one ends the work, so the
/// resolver now asks both; this is the one that shutting down actually controls.
///
/// [`sleep`](Cancel::sleep) is the reason it carries a `Notify` rather than being
/// a bare flag: a substrate's backoff widens to half a minute, and a session that has
/// ended should not wait out the rest of one.
#[derive(Debug, Clone, Default)]
pub(crate) struct Cancel(Arc<CancelInner>);

#[derive(Debug, Default)]
struct CancelInner {
    stopped: AtomicBool,
    woken: Notify,
}

impl Cancel {
    /// End the session's background work. Idempotent.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::Relaxed);
        self.0.woken.notify_waiters();
    }

    pub fn stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Relaxed)
    }

    /// Pend until the session ends — what a loop races against its own work,
    /// where [`sleep`](Cancel::sleep) is what a backoff waits out.
    pub async fn stopped_wait(&self) {
        // Registered before the check, for the same race `sleep` closes.
        let woken = self.0.woken.notified();
        if self.stopped() {
            return;
        }
        // Only `stop` notifies, and it sets the flag first.
        woken.await;
    }

    /// Sleep, unless the session ends first — `false` if it did.
    pub async fn sleep(&self, duration: Duration) -> bool {
        // Registered before the check, so a `stop` racing this cannot slip between
        // the two and leave the sleeper waiting on a notification already sent.
        let woken = self.0.woken.notified();
        if self.stopped() {
            return false;
        }
        n0_future::future::race(
            async {
                n0_future::time::sleep(duration).await;
                true
            },
            async {
                woken.await;
                false
            },
        )
        .await
    }
}
