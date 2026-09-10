//! The layer carry's decisions (§16.11, §16.12): when a press engages, what it
//! previews, and what its release lays down.
//!
//! Its two halves arrive out of order. The pointer's travel is known at once, but the
//! layer under the press is a GPU readback, so a flick can be over before the hit test
//! answers — which is why a [`Carry`] outlives its release and [`Carry::settled`]
//! answers only once both are in. What a frontend keeps is the readback, the preview it
//! has on screen and the commands that change them.

use stark_engine::ObservableState;
use stark_model::document::LayerId;
use stark_model::geom::{IVec2, Vec2};

use crate::input::PointerKind;

/// How far a mouse press has to travel before a carry engages, in screen px.
pub const MOUSE_DEADZONE: f32 = 4.0;

/// How far a pen or finger press has to travel, in screen px.
///
/// Wider than a mouse's because a pen tip flexes and a fingertip rolls on the way off
/// the glass, and of the two mistakes the tap is the one to protect: a nudge that did
/// not happen is retried in a second, one that did leaves the painting changed.
pub const PEN_DEADZONE: f32 = 10.0;

/// How far a press made with `kind` has to travel before the carry engages.
///
/// Screen px, because what it separates is a tap of the hand from a drag of it, and
/// neither becomes the other by zooming — so zooming in is the escape hatch from a
/// deadzone too coarse for the nudge wanted.
pub fn deadzone(kind: PointerKind) -> f32 {
    match kind {
        PointerKind::Mouse => MOUSE_DEADZONE,
        PointerKind::Pen | PointerKind::Touch => PEN_DEADZONE,
    }
}

/// Which layer a press landed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// The hit test has not answered yet.
    Pending,
    /// This layer's paint, whose frame stood at `base` when it was picked (§14.12).
    /// Latched then, because mid-drag the projection reports the *previewed* frame, and
    /// a base that followed it would compound.
    Layer { id: LayerId, base: IVec2 },
    /// Nothing the canvas shows is under the press. The gesture stays in flight and
    /// does nothing.
    Nothing,
}

/// What a finished carry lays down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settle {
    /// One logged translation of `layer`'s frame to `to`, superseding the preview.
    Commit { layer: LayerId, to: IVec2 },
    /// Nothing, and whatever preview is up comes down: a tap, whose whole act was
    /// selecting the layer; a drag that came back to the press; nothing under it.
    Nothing,
}

/// Where a layer carry has got to, and what it is waiting for (§16.11).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Carry {
    /// Which press this is, so an answer to a press since replaced — a double-click
    /// racing its own readback — is dropped rather than written into its successor.
    press: u64,
    /// Where the press landed and where the pointer is, canvas px. Measured from the
    /// press rather than the last move, so a long drag cannot drift.
    from: Vec2,
    to: Vec2,
    /// Where the press landed, screen px, and how far from there it must travel.
    /// Latched at the press: a threshold that moved mid-drag could be crossed by a hand
    /// standing still.
    from_screen: Vec2,
    deadzone: f32,
    /// Whether the pointer has been past the deadzone. Latched: a drag that wanders back
    /// to the press is still a drag, asking to put the layer back.
    dragged: bool,
    hit: Hit,
    /// A selection pinned the press to the active layer, so the first travel floats the
    /// selected paint into a child layer and the drag carries that (§16.12).
    pinned: bool,
    floated: bool,
    released: bool,
}

impl Carry {
    /// Open carry number `press` at `from` (canvas px) and `from_screen` (screen px).
    ///
    /// `pinned` is [`pinned_layer`]'s answer: a selection in force answers the press
    /// without asking the canvas.
    pub fn press(
        press: u64,
        from: Vec2,
        from_screen: Vec2,
        deadzone: f32,
        pinned: Option<Hit>,
    ) -> Self {
        Self {
            press,
            from,
            to: from,
            from_screen,
            deadzone,
            dragged: false,
            hit: pinned.unwrap_or(Hit::Pending),
            pinned: pinned.is_some(),
            floated: false,
            released: false,
        }
    }

    /// Whether this carry is waiting for the hit test on press `press` — `false` for a
    /// press since replaced, and once answered.
    pub fn awaits(&self, press: u64) -> bool {
        self.press == press && self.hit == Hit::Pending
    }

    /// The hit test answered `hit`. Ignored unless one was still pending.
    pub fn answered(&mut self, hit: Hit) {
        if self.hit == Hit::Pending {
            self.hit = hit;
        }
    }

    /// The pointer moved to `to` (canvas px), `at` (screen px). Ignored once released: a
    /// captured pointer can deliver a move after its own release.
    pub fn moved(&mut self, to: Vec2, at: Vec2) {
        if self.released {
            return;
        }
        self.to = to;
        self.dragged |= at.distance(self.from_screen) >= self.deadzone;
    }

    /// The pointer is up. What is left is waiting for the answer to the press.
    pub fn released(&mut self) {
        self.released = true;
    }

    /// Whether the pointer is up.
    pub fn is_released(&self) -> bool {
        self.released
    }

    /// The layer whose selected paint has to be floated before this carry moves
    /// anything: a pinned press that has begun to travel, once (§16.12).
    pub fn float_due(&self) -> Option<LayerId> {
        match self.hit {
            Hit::Layer { id, .. } if self.pinned && !self.floated && self.delta() != Vec2::ZERO => {
                Some(id)
            }
            _ => None,
        }
    }

    /// The float was made, and `child` is what it left to carry ([`float_child`]).
    pub fn floated(&mut self, child: Hit) {
        self.floated = true;
        self.hit = child;
    }

    /// The frame to preview now: the layer and where its frame is carried to. `None` for
    /// no preview — which after a drag back to the press is the layer shown where it was.
    pub fn wanted(&self) -> Option<(LayerId, IVec2)> {
        if self.float_due().is_some() {
            return None;
        }
        let Hit::Layer { id, base } = self.hit else {
            return None;
        };
        let delta = self.delta();
        (delta != Vec2::ZERO).then(|| (id, base + delta.as_ivec2()))
    }

    /// What to lay down, once **both** halves are in: the pointer is up and the hit test
    /// has answered. `None` until then.
    pub fn settled(&self) -> Option<Settle> {
        if !self.released || self.hit == Hit::Pending {
            return None;
        }
        Some(match self.wanted() {
            Some((layer, to)) => Settle::Commit { layer, to },
            None => Settle::Nothing,
        })
    }

    /// The translation asked for, in **whole canvas pixels**.
    ///
    /// A translation by whole texels resamples nothing (§16.4), where a fractional one
    /// costs the layer a generation of blur for a movement no eye asked for. Nothing
    /// before the deadzone, which is what makes the tap and the carry one gesture.
    fn delta(&self) -> Vec2 {
        if !self.dragged {
            return Vec2::ZERO;
        }
        (self.to - self.from).round()
    }
}

/// The layer a press must carry because a **selection** says so, or `None` where no mask
/// is in force and the press is free to go looking (§16.11).
///
/// A mask was drawn against paint on a layer the artist had in mind, and a press that
/// re-targeted would cut a different layer's paint through their lasso. A universal
/// selection is not one. [`Hit::Nothing`] when the pinned layer holds no paint: the mask
/// still forbids looking elsewhere.
pub fn pinned_layer(o: &ObservableState) -> Option<Hit> {
    if !o.has_selection {
        return None;
    }
    Some(
        match o
            .layers
            .iter()
            .find(|l| l.id == o.active_layer && l.is_paintable())
        {
            Some(l) => Hit::Layer {
                id: l.id,
                base: l.translation,
            },
            None => Hit::Nothing,
        },
    )
}

/// What a float of `parent`'s selection left to carry: the child it made active, from
/// where its frame starts — or [`Hit::Nothing`] where the engine declined, the mask
/// holding nothing of that layer.
pub fn float_child(o: &ObservableState, parent: LayerId) -> Hit {
    if o.active_layer == parent {
        return Hit::Nothing;
    }
    Hit::Layer {
        id: o.active_layer,
        base: layer_translation(o, o.active_layer),
    }
}

/// Where `id`'s frame stands, per the projection — zero for a layer the roster does not
/// hold, which the engine will then refuse to move.
pub fn layer_translation(o: &ObservableState, id: LayerId) -> IVec2 {
    o.layers
        .iter()
        .find(|l| l.id == id)
        .map_or(IVec2::ZERO, |l| l.translation)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYER: LayerId = LayerId::solo(7);
    const BASE: IVec2 = IVec2::new(30, -12);

    /// A mouse carry pressed at the origin, not yet answered.
    fn pressed() -> Carry {
        Carry::press(1, Vec2::ZERO, Vec2::ZERO, MOUSE_DEADZONE, None)
    }

    fn hit() -> Hit {
        Hit::Layer {
            id: LAYER,
            base: BASE,
        }
    }

    /// Move to `(x, y)` with the canvas and the screen agreeing.
    fn drag(carry: &mut Carry, x: f32, y: f32) {
        carry.moved(Vec2::new(x, y), Vec2::new(x, y));
    }

    /// A tap selects the layer under it and lays nothing down.
    #[test]
    fn a_tap_commits_nothing() {
        let mut carry = pressed();
        drag(&mut carry, MOUSE_DEADZONE * 0.5, 0.0);
        carry.answered(hit());
        carry.released();
        assert_eq!(carry.wanted(), None);
        assert_eq!(carry.settled(), Some(Settle::Nothing));
    }

    /// A drag that comes back to the press shows the layer put back and commits nothing
    /// — but it stays a drag, so a pixel's nudge from there is still a nudge.
    #[test]
    fn a_drag_back_to_the_press_puts_it_back() {
        let mut carry = pressed();
        carry.answered(hit());
        drag(&mut carry, 20.0, 0.0);
        assert_eq!(carry.wanted(), Some((LAYER, BASE + IVec2::new(20, 0))));
        drag(&mut carry, 1.0, 0.0);
        assert_eq!(
            carry.wanted(),
            Some((LAYER, BASE + IVec2::new(1, 0))),
            "latched past the deadzone"
        );
        drag(&mut carry, 0.0, 0.0);
        assert_eq!(carry.wanted(), None);
        carry.released();
        assert_eq!(carry.settled(), Some(Settle::Nothing));
    }

    /// A flick can be over before the hit test answers; the release waits, and the answer
    /// still lays the carry down.
    #[test]
    fn an_answer_after_the_release_still_commits() {
        let mut carry = pressed();
        drag(&mut carry, 20.0, 5.0);
        carry.released();
        assert_eq!(carry.settled(), None, "waiting on the hit test");
        carry.answered(hit());
        assert_eq!(
            carry.settled(),
            Some(Settle::Commit {
                layer: LAYER,
                to: BASE + IVec2::new(20, 5)
            })
        );
    }

    /// An answer is for the press that asked: a later press's record is not waiting on it,
    /// and a record takes one answer.
    #[test]
    fn a_stale_press_is_ignored() {
        let mut carry = Carry::press(2, Vec2::ZERO, Vec2::ZERO, MOUSE_DEADZONE, None);
        assert!(!carry.awaits(1), "the answer to a replaced press");
        assert!(carry.awaits(2));
        carry.answered(Hit::Nothing);
        assert!(!carry.awaits(2), "already answered");
        carry.answered(hit());
        drag(&mut carry, 20.0, 0.0);
        assert_eq!(carry.wanted(), None, "the first answer stands");
    }

    /// A selection pins the press, and its first travel floats the selected paint — once;
    /// the drag carries the float from then on.
    #[test]
    fn a_pinned_press_floats_exactly_once() {
        let mut carry = Carry::press(1, Vec2::ZERO, Vec2::ZERO, MOUSE_DEADZONE, Some(hit()));
        assert!(!carry.awaits(1), "a selection answers without the canvas");
        assert_eq!(carry.float_due(), None, "nothing travelled");
        drag(&mut carry, 20.0, 0.0);
        assert_eq!(carry.float_due(), Some(LAYER));
        assert_eq!(carry.wanted(), None, "nothing is carried before the float");
        let child = LayerId::solo(8);
        carry.floated(Hit::Layer {
            id: child,
            base: IVec2::ZERO,
        });
        assert_eq!(carry.float_due(), None);
        drag(&mut carry, 40.0, 0.0);
        assert_eq!(carry.float_due(), None);
        assert_eq!(carry.wanted(), Some((child, IVec2::new(40, 0))));
    }

    /// The carry moves in whole canvas pixels, which resample nothing (§16.4).
    #[test]
    fn the_delta_is_whole_pixels() {
        let mut carry = pressed();
        carry.answered(hit());
        drag(&mut carry, 10.4, -7.6);
        assert_eq!(carry.wanted(), Some((LAYER, BASE + IVec2::new(10, -8))));
    }

    /// A finger engages as late as a pen does, and both later than a mouse.
    #[test]
    fn a_finger_engages_as_late_as_a_pen() {
        assert_eq!(deadzone(PointerKind::Touch), deadzone(PointerKind::Pen));
        assert!(deadzone(PointerKind::Pen) > deadzone(PointerKind::Mouse));
    }
}
