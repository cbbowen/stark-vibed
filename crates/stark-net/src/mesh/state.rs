//! Per-origin delivery bookkeeping: deduplication and gap detection.
//!
//! Flooding means the same payload arrives over every path that exists, so
//! *every* peer must recognise what it has already seen — otherwise messages
//! circulate forever. Each payload is stamped `(origin, seq)` with a per-origin
//! counter, which gives dedup and, for free, the ability to notice we never
//! received something (iroh-gossip's `Lagged`).

use std::collections::{BTreeSet, HashMap};

use super::transport::PeerId;

/// What to do with a just-received payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Delivery {
    /// First time we've seen it: deliver it and forward it on.
    pub fresh: bool,
    /// A hole in this origin's sequence got too old to wait for — messages were
    /// lost. Callers surface this so the session can resync.
    pub lagged: bool,
}

#[derive(Debug, Default)]
struct Origin {
    /// Every seq at or below this has been delivered, or written off as lost.
    watermark: u64,
    /// Delivered seqs above the watermark (arrived out of order).
    ahead: BTreeSet<u64>,
}

/// Tracks, for each origin, which sequence numbers have been delivered.
#[derive(Debug)]
pub(crate) struct DeliveryTracker {
    origins: HashMap<PeerId, Origin>,
    /// How far past the watermark a hole may sit before we give up on it.
    lag_threshold: u64,
}

impl DeliveryTracker {
    pub fn new(lag_threshold: u64) -> Self {
        Self {
            origins: HashMap::new(),
            lag_threshold: lag_threshold.max(1),
        }
    }

    /// Record `(origin, seq)`; report whether it is new and whether we just gave
    /// up on missing messages.
    pub fn observe(&mut self, origin: PeerId, seq: u64) -> Delivery {
        let state = self.origins.entry(origin).or_default();

        // Sequences start at 1, so a watermark of 0 means "nothing yet".
        if seq == 0 || seq <= state.watermark || state.ahead.contains(&seq) {
            return Delivery {
                fresh: false,
                lagged: false,
            };
        }

        state.ahead.insert(seq);
        // Absorb the run of contiguous sequences now sitting on the watermark.
        while state.ahead.remove(&(state.watermark + 1)) {
            state.watermark += 1;
        }

        // A hole this far behind the frontier is not going to be filled.
        let mut lagged = false;
        if let Some(&highest) = state.ahead.iter().next_back()
            && highest.saturating_sub(state.watermark) > self.lag_threshold
        {
            let resume = state
                .ahead
                .iter()
                .next()
                .copied()
                .expect("non-empty while highest exists");
            state.watermark = resume.saturating_sub(1);
            while state.ahead.remove(&(state.watermark + 1)) {
                state.watermark += 1;
            }
            lagged = true;
        }

        Delivery {
            fresh: true,
            lagged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(n: u8) -> PeerId {
        PeerId([n; 32])
    }

    #[test]
    fn first_delivery_is_fresh_and_repeats_are_not() {
        let mut t = DeliveryTracker::new(128);
        assert!(t.observe(peer(1), 1).fresh);
        assert!(!t.observe(peer(1), 1).fresh);
        assert!(!t.observe(peer(1), 1).fresh);
    }

    #[test]
    fn origins_are_tracked_independently() {
        let mut t = DeliveryTracker::new(128);
        assert!(t.observe(peer(1), 1).fresh);
        // Same sequence number, different origin: still new.
        assert!(t.observe(peer(2), 1).fresh);
    }

    #[test]
    fn out_of_order_arrivals_all_deliver_exactly_once() {
        let mut t = DeliveryTracker::new(128);
        for seq in [3u64, 1, 2, 5, 4] {
            assert!(t.observe(peer(1), seq).fresh, "seq {seq} should be fresh");
        }
        for seq in 1..=5 {
            assert!(!t.observe(peer(1), seq).fresh, "seq {seq} should be a dup");
        }
    }

    #[test]
    fn a_hole_that_fills_does_not_report_lag() {
        let mut t = DeliveryTracker::new(4);
        assert!(!t.observe(peer(1), 1).lagged);
        assert!(!t.observe(peer(1), 3).lagged); // gap at 2
        assert!(!t.observe(peer(1), 2).lagged); // filled in time
    }

    #[test]
    fn a_hole_that_never_fills_reports_lag_once_and_moves_on() {
        let mut t = DeliveryTracker::new(4);
        assert!(t.observe(peer(1), 1).fresh);
        // 2 never arrives; the frontier runs away from it.
        let mut lagged = 0;
        for seq in 3..=9 {
            if t.observe(peer(1), seq).lagged {
                lagged += 1;
            }
        }
        assert_eq!(lagged, 1, "lag should be reported once per hole");
        // Having given up, the late arrival is not resurrected as fresh.
        assert!(!t.observe(peer(1), 2).fresh);
        // And the frontier keeps working normally afterwards.
        assert!(t.observe(peer(1), 10).fresh);
    }

    #[test]
    fn sequence_zero_is_never_delivered() {
        let mut t = DeliveryTracker::new(128);
        assert!(!t.observe(peer(1), 0).fresh);
    }
}
