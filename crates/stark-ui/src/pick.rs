//! The eyedropper as a frontend holds it (§18.0.2): what a sample is taken with,
//! how far its reach resolves, and whether a press right now would take one.
//!
//! [`Engine::pick_color`](stark_engine::Engine::pick_color) is a **request** and
//! these are its arguments (§4) — nothing in the engine reads them between calls, so
//! a copy projected back through `observe()` would be state with no owner. What is
//! *not* the chrome's is the resolution from a choice to a
//! [`PickSource`]: which layer "this layer" means is settled at the moment of the
//! sample, against whichever layer is selected then, so a bar cannot be left holding
//! the id of a layer that has since been deleted.
//!
//! The arming half is here for [`input`](crate::input)'s reason. The eyedropper is
//! not a tool you switch to — the chord over the brush *is* the binding — so "is it
//! armed" is a question the drag table answers plus a handful of stand-downs only the
//! chrome knows about, and two frontends asking it two ways would promise the sample
//! in one and take it in the other.

use stark_engine::{PickOptions, PickSource};
use stark_model::document::LayerId;

use crate::commands::PickScope;

/// The patches the bars offer, as **radii** — the half-width of the averaged square,
/// which is what [`PickOptions::radius`] takes. `2r + 1` is the square each describes
/// ([`patch_word`]), so the words a bar wears are derived rather than written beside
/// the numbers they have to agree with.
pub const PATCHES: [u32; 4] = [0, 1, 2, 5];

/// How a patch is spelled on a bar: the prior art's point sample, or the `N×N` square
/// a radius describes.
pub fn patch_word(radius: u32) -> String {
    if radius == 0 {
        "Point".to_string()
    } else {
        let n = 2 * radius + 1;
        format!("{n}\u{00D7}{n}")
    }
}

/// What the eyedropper's options bar holds: the two questions a sample answers to,
/// and how much canvas one averages.
///
/// A value rather than three, because they are one gesture's worth of settings and a
/// call site passing them positionally could transpose the fence and the reach. How
/// each is *stored* is still each frontend's — a signal on the web, a field natively
/// — which is the crate's own rule (§11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampler {
    /// How far up the stack a sample sees.
    pub scope: PickScope,
    /// Whether the reach is confined to the selected layer's **group** — its siblings
    /// and the layer carrying them (§14.2). On by default: sampling near paint
    /// usually means sampling the passage being worked, not whatever other group
    /// happens to show through at that point.
    pub group_only: bool,
    /// Half-width of the averaged square, in canvas px (0 = point sample).
    pub radius: u32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            scope: PickScope::default(),
            group_only: true,
            radius: 0,
        }
    }
}

impl Sampler {
    /// What the engine is asked for, resolved against `active` — the layer selected
    /// at the moment of the sample.
    ///
    /// A document with no layer selected falls back to the whole document rather than
    /// sampling nothing. The canvas color stands behind the sample exactly when the
    /// group fence is down: a group is paint, and the whole document is a picture on
    /// a canvas (§15.5).
    ///
    /// **One arm per reach**, each answering the layer and the fence for itself. The
    /// resolution used to be a `match` on the pair with two trailing wildcards, which
    /// meant a fourth [`PickScope`] would compile and silently sample the whole
    /// composite — and this module's own note says the resolution is not the chrome's
    /// to get wrong. The scope is a closed set everywhere else it is met
    /// (`PickScope::VARIANTS` builds both bars), so it is closed here too.
    pub fn options(self, active: Option<LayerId>) -> PickOptions {
        let source = match self.scope {
            PickScope::ThisLayer => match active {
                // One layer alone is one layer alone whichever side of the fence it is
                // asked from: the group is what a *reach* runs through, and this reach
                // is one layer.
                Some(id) => PickSource::Layer(id),
                None => self.whole_document(),
            },
            PickScope::AndBelow => match active {
                Some(id) if self.group_only => PickSource::Group {
                    layer: id,
                    below: true,
                },
                Some(id) => PickSource::Below(id),
                None => self.whole_document(),
            },
            PickScope::AllLayers => match active {
                Some(id) if self.group_only => PickSource::Group {
                    layer: id,
                    below: false,
                },
                // Unfenced, "every visible layer" already *is* the whole document, so
                // the selected layer decides nothing and the arm falls through with
                // the layerless one.
                _ => self.whole_document(),
            },
        };
        PickOptions {
            source,
            radius: self.radius,
        }
    }

    /// The two answers that name no layer — what every reach falls back to with none
    /// selected, and what "all layers" is outright.
    ///
    /// The fence is the whole of the choice: on, a group answers and bare canvas
    /// answers nothing; off, the document answers **over the substrate**, which is the
    /// one reach whose answer can be a color no layer holds (§15.5).
    fn whole_document(self) -> PickSource {
        if self.group_only {
            PickSource::Composite
        } else {
            PickSource::CompositeOverSubstrate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A patch's word is its radius spelled out, so the bars cannot offer a "5×5"
    /// that averages some other square.
    #[test]
    fn a_patch_is_named_by_its_radius() {
        assert_eq!(patch_word(0), "Point");
        assert_eq!(patch_word(1), "3\u{00D7}3");
        assert_eq!(patch_word(5), "11\u{00D7}11");
    }

    /// The shipped patches are ordered by how much canvas each takes in, which is
    /// what makes the row one question rather than four buttons — the same claim
    /// `PickScope::VARIANTS`'s ordering makes.
    #[test]
    fn the_patches_widen() {
        assert!(PATCHES.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(PATCHES[0], 0, "the row leads with the single texel");
    }

    /// The fence is what puts the canvas color behind a sample: on, a group answers
    /// and bare canvas answers nothing; off, the document answers over the substrate
    /// (§15.5). With no layer selected there is no group, and the two ends of the
    /// fence are the two composites.
    #[test]
    fn the_canvas_stands_behind_a_sample_exactly_when_the_fence_is_down() {
        let open = Sampler {
            group_only: false,
            ..Sampler::default()
        };
        assert_eq!(
            Sampler::default().options(None).source,
            PickSource::Composite
        );
        assert_eq!(
            open.options(None).source,
            PickSource::CompositeOverSubstrate
        );
    }

    /// Every reach resolves against the layer selected *now*, and the fence chooses
    /// between the group and the plain reach at each of them.
    #[test]
    fn each_reach_resolves_against_the_selected_layer() {
        let id = LayerId::ROOT;
        let fenced = Sampler::default();
        let open = Sampler {
            group_only: false,
            ..fenced
        };
        for (scope, fenced_source, open_source) in [
            (
                PickScope::ThisLayer,
                PickSource::Layer(id),
                PickSource::Layer(id),
            ),
            (
                PickScope::AndBelow,
                PickSource::Group {
                    layer: id,
                    below: true,
                },
                PickSource::Below(id),
            ),
            (
                PickScope::AllLayers,
                PickSource::Group {
                    layer: id,
                    below: false,
                },
                // The whole document *over the substrate*: with the fence down the
                // canvas color is in the question, which is the one reach whose
                // answer can be a color no layer holds (§15.5).
                PickSource::CompositeOverSubstrate,
            ),
        ] {
            assert_eq!(
                Sampler { scope, ..fenced }.options(Some(id)).source,
                fenced_source,
                "{scope:?} fenced"
            );
            assert_eq!(
                Sampler { scope, ..open }.options(Some(id)).source,
                open_source,
                "{scope:?} unfenced"
            );
        }
    }

    /// One layer alone is one layer alone whichever side of the fence it is asked
    /// from: the group is what a *reach* runs through, and this reach is one layer.
    #[test]
    fn one_layer_does_not_care_about_the_fence() {
        let id = LayerId::ROOT;
        let this = Sampler {
            scope: PickScope::ThisLayer,
            ..Sampler::default()
        };
        assert_eq!(
            this.options(Some(id)).source,
            Sampler {
                group_only: false,
                ..this
            }
            .options(Some(id))
            .source
        );
    }

    /// The radius is carried through untouched — the engine clamps it, which is where
    /// a bound on what one sample may cost belongs.
    #[test]
    fn the_patch_reaches_the_engine() {
        for radius in PATCHES {
            let sampler = Sampler {
                radius,
                ..Sampler::default()
            };
            assert_eq!(sampler.options(None).radius, radius);
        }
    }
}
