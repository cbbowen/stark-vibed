//! Timeline mode's transport (§18.2.4): how fast playback walks the history, and how
//! many marks a track draws for it.
//!
//! The playhead is not here, and not in either frontend: it is the engine timeline's
//! applied/withheld split (`Timeline::seek`), the same one undo and redo move. What is
//! here is the pace a transport steps it at.

/// Actions per second at 1×. Eight is about the rate at which a painting reads as being
/// *made* rather than as a slideshow of states: fast enough that a session's worth of
/// strokes lands in a minute or two, slow enough that a single one can be seen arriving.
pub const BASE_RATE: f32 = 8.0;

/// The shortest wait worth asking a timer for. Below one frame the timer stops being
/// what paces playback — a browser clamps it, and the rate silently stops responding to
/// the speed control — so past this point the *stride* grows instead ([`pace`]).
pub const MIN_TICK_MS: f32 = 16.0;

/// Most actions a tick crosses: past any real history, and nowhere near overflowing `at + stride`.
pub const MAX_STRIDE: usize = 1 << 20;

/// Most ticks a track draws. Past this they sit closer together than a hairline and stop
/// being marks at all; the fill and the counter still say where the playhead is.
pub const MAX_TICKS: usize = 240;

/// The playback rates a transport offers, as multiples of [`BASE_RATE`], with the word
/// each wears.
pub const SPEEDS: [(f32, &str); 5] = [
    (0.25, "\u{00BC}\u{00D7}"),
    (0.5, "\u{00BD}\u{00D7}"),
    (1.0, "1\u{00D7}"),
    (2.0, "2\u{00D7}"),
    (4.0, "4\u{00D7}"),
];

/// How long to wait between steps (ms), and how many actions to cross each time, at
/// `speed` times [`BASE_RATE`].
///
/// One step per tick until a tick would be shorter than [`MIN_TICK_MS`]; from there the
/// interval is pinned and the stride takes over. Without that, a speed past the timer's
/// floor would be indistinguishable from a slower one — the control would still move and
/// nothing would happen.
pub fn pace(speed: f32) -> (i32, usize) {
    let per_step = 1000.0 / (BASE_RATE * speed.max(0.01));
    if per_step >= MIN_TICK_MS {
        (per_step.round() as i32, 1)
    } else {
        (
            MIN_TICK_MS as i32,
            (MIN_TICK_MS / per_step)
                .round()
                .clamp(1.0, MAX_STRIDE as f32) as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Speeds from well under the slowest chip to far past the timer's floor, five
    /// percent apart — fine enough that every rounding step of the interval is crossed,
    /// which is where a slip in monotonicity would hide.
    fn sweep() -> impl Iterator<Item = f32> {
        (0..=200).map(|i| 0.1 * 1.05f32.powi(i))
    }

    /// Actions per second at `speed`, as a transport actually plays it.
    fn rate(speed: f32) -> f32 {
        let (ms, stride) = pace(speed);
        stride as f32 * 1000.0 / ms as f32
    }

    /// A faster speed never waits longer between ticks, and no speed waits less than the
    /// floor it exists to respect.
    #[test]
    fn a_faster_speed_never_waits_longer() {
        for (slow, fast) in sweep().zip(sweep().skip(1)) {
            let (was, now) = (pace(slow).0, pace(fast).0);
            assert!(
                now <= was,
                "{fast}\u{00D7} waits {now} ms, longer than {slow}\u{00D7}'s {was} ms"
            );
            assert!(
                now >= MIN_TICK_MS as i32,
                "{fast}\u{00D7} asks the timer for {now} ms, under the floor"
            );
        }
    }

    /// Every tick crosses at least one action and at most [`MAX_STRIDE`] — including at
    /// speeds no chip offers: nothing, less than nothing, not a number, unbounded.
    #[test]
    fn every_stride_is_at_least_one_action_and_bounded() {
        for speed in sweep().chain([0.0, -1.0, f32::NAN, f32::INFINITY]) {
            let (_, stride) = pace(speed);
            assert!(
                (1..=MAX_STRIDE).contains(&stride),
                "{speed}\u{00D7} strides {stride} actions a tick"
            );
        }
    }

    /// Past the floor the interval is pinned, and a faster speed still plays faster —
    /// the whole reason the stride exists. And every chip the transport offers plays at
    /// a rate of its own, so no chip is one that moves and changes nothing.
    #[test]
    fn past_the_floor_a_faster_speed_still_plays_faster() {
        // The speed at which one step a tick takes exactly the floor.
        let floor = 1000.0 / (BASE_RATE * MIN_TICK_MS);
        for (slow, fast) in [
            (floor, 4.0 * floor),
            (2.0 * floor, 4.0 * floor),
            (4.0 * floor, 8.0 * floor),
        ] {
            assert_eq!(
                pace(fast).0,
                MIN_TICK_MS as i32,
                "{fast}\u{00D7} is past the floor, so its interval is the floor"
            );
            assert!(
                rate(fast) > rate(slow),
                "{fast}\u{00D7} plays {} actions/s, no faster than {slow}\u{00D7}'s {}",
                rate(fast),
                rate(slow)
            );
        }
        for ((slow, a), (fast, b)) in SPEEDS.iter().zip(SPEEDS.iter().skip(1)) {
            assert!(
                rate(*fast) > rate(*slow),
                "the {b} chip plays no faster than the {a} chip"
            );
        }
    }
}
