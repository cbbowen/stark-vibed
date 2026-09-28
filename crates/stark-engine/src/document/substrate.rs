//! Which canvas substrate the document stands on, and how large it is laid (§6.4).
//!
//! The *derived* half of the pair, and the smaller one: the log carries a
//! [`SubstrateId`] and a [`SubstrateScale`] through two separate actions, and this is
//! the pair they compose into — [`DocState`](super::state::DocState)'s field, the thing
//! a renderer is handed and the key a bake is registered under.
//!
//! **Here rather than beside the bake** (`gpu::substrate`), because it holds no texture
//! and answers no shader: it is two document facts in one value, and `document/` is
//! below `gpu/` (§2). What the pair *means to a pass* — the grain's tile size, the pitch
//! the deposition tooth reads the map through — stays with the bake, as a second
//! inherent `impl` over there. Reading which module a method is in tells you which half
//! you are in, exactly as it does for `layer.rs` and `selection.rs`.

use stark_model::{SubstrateId, SubstrateScale};

/// **A canvas substrate: which map, and how large it is laid** (§6.4).
///
/// The pair, not the id alone, because the map that gets baked is a function of both:
/// the rise a tip meets is a difference taken across `tooth::TOOTH_REACH` *canvas px*
/// expressed in the map's own texels (`tooth::pack_substrate`), so laying the same
/// substrate at twice the size halves that span and changes what the tooth bites.
/// Baking one map and scaling the lookup would be the compensating fudge §1 rules out:
/// it would report the rise over six px as if it were the rise over three.
///
/// So this is the registry's key, while the *bytes* stay keyed by the [`SubstrateId`]
/// alone (`gpu::registry`): one height map, a bake per scale it is laid at.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Substrate {
    /// Which substrate — the document's `SubstrateId`, the thing the bytes are named by.
    pub id: SubstrateId,
    /// How large it is laid, as the document says (`DocState::substrate_scale`).
    pub scale: SubstrateScale,
}

impl Substrate {
    /// The substrate `id` at natural size — what an engine with no document opinion
    /// stands on, and what every caller that has only an id means.
    pub fn new(id: SubstrateId) -> Self {
        Self {
            id,
            scale: SubstrateScale::NATURAL,
        }
    }
}
