//! The engine as both frontends hold it (§4, §11.2): the [`Engine`], the shipped assets
//! it has loaded by catalog name, and the few acts on it that are more than a forward.
//!
//! Everything else is the engine's own method, reached through [`Desk::engine`] and
//! [`Desk::engine_mut`]. A wrapper per engine method was two copies of one API, one
//! per frontend.
//!
//! **Replacing the document has one door**, [`Desk::replace`], because a replacement
//! owes more than the engine call: the content the new document names goes in first,
//! and the view has to be put on the piece after (§18.1.2). Each consumer's
//! `clippy.toml` refuses the engine's replacement methods by path, and
//! `tests/one_way_to_replace.rs` reads the source for the crate CI does not lint.

use stark_engine::command::ViewCommand;
use stark_engine::{Engine, EngineError, Extent2, Identity, ObservableState, Transfer};
use stark_model::{AssetId, AssetNeed, ColorSpaceId, DocError, DocumentFile, SubstrateId};

use crate::prefs::Hdr;

/// One frontend's engine, and what it knows about the catalog (§6.4, §6.6).
pub struct Desk {
    engine: Engine,
    /// How the surface reads the engine's texels (§6.5), fixed with the format the
    /// engine was built for.
    transfer: Transfer,
    shapes: Loaded<AssetId>,
    substrates: Loaded<SubstrateId>,
}

/// What a document is replaced with.
#[derive(Clone, Copy, Debug)]
pub enum Replacement<'a> {
    /// A file's log, replayed with its whole history (§8).
    Open(&'a DocumentFile),
    /// A session's log, authored from here on as `Identity` (§12.4).
    Join(&'a DocumentFile, Identity),
    /// An empty document in a color space, on a substrate (§6.7).
    New(ColorSpaceId, SubstrateId),
}

/// What [`Desk::replace`] made of a replacement it went through with.
#[derive(Debug)]
#[must_use = "a skipped need is content the new document still names"]
pub struct Replaced {
    /// The new document's projection, with the view already on its piece.
    pub seen: ObservableState,
    /// Owed content a join could not install, left to the peer fetch
    /// (`stark_net::Joined::owed`). Always empty for an open or a new document, which
    /// refuse instead.
    pub skipped: Vec<(AssetNeed, EngineError)>,
}

impl Desk {
    /// Hold `engine`, which renders for a surface read in `transfer`.
    pub fn new(engine: Engine, transfer: Transfer) -> Self {
        Self {
            engine,
            transfer,
            shapes: Loaded::default(),
            substrates: Loaded::default(),
        }
    }

    /// A second desk on this one's device and shared state (`Engine::new_sharing`),
    /// around a document of its own.
    ///
    /// The name indices come along: the ids are content-addressed and the assets behind
    /// them are shared, so this desk's answers are the donor's.
    pub fn sharing(&self, viewport: Extent2) -> Self {
        Self {
            engine: Engine::new_sharing(&self.engine, viewport),
            transfer: self.transfer,
            shapes: self.shapes.clone(),
            substrates: self.substrates.clone(),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// How the surface reads the engine's texels (§6.5).
    pub fn transfer(&self) -> Transfer {
        self.transfer
    }

    /// Whether the surface can show anything above white (§6.5). A wide gamut is not
    /// range, so a `display-p3` surface answers no.
    pub fn hdr_capable(&self) -> bool {
        self.transfer.is_hdr()
    }

    /// Tell the engine what the display is (§6.5): [`crate::lighting::output`] over
    /// `choice`, this surface's transfer, and the display's own headroom where the
    /// platform states one.
    pub fn apply_output(&mut self, choice: Hdr, display_headroom: Option<f32>) {
        let output = crate::lighting::output(choice, self.transfer, display_headroom);
        self.engine.process(ViewCommand::SetOutput(output));
    }

    /// Import a shipped brush shape's bytes and remember its id under `name`.
    pub fn load_builtin_shape(
        &mut self,
        name: &'static str,
        png: &[u8],
    ) -> stark_engine::Result<AssetId> {
        let id = self.engine.import_brush(png)?;
        self.shapes.record(name, id);
        Ok(id)
    }

    /// The shipped shape `name` resolved to, or `None` until it has been loaded.
    pub fn builtin_shape(&self, name: &str) -> Option<AssetId> {
        self.shapes.get(name)
    }

    /// Import a shipped substrate's height map and remember its id under `name`.
    pub fn load_builtin_substrate(
        &mut self,
        name: &'static str,
        png: &[u8],
    ) -> stark_engine::Result<SubstrateId> {
        let id = self.engine.import_substrate(png)?;
        self.substrates.record(name, id);
        Ok(id)
    }

    /// The shipped substrate `name` resolved to, or `None` until it has been loaded.
    pub fn builtin_substrate(&self, name: &str) -> Option<SubstrateId> {
        self.substrates.get(name)
    }

    /// Install content under the need that asked for it — read out of this build's
    /// bundle or arrived off a peer (§12.4).
    ///
    /// Refused, in every arm, when the bytes are not what the id names
    /// (`DocError::Misnamed`). A brush's id comes out of its import, so a mismatched one
    /// is held under its own id and the need stays unmet. **Exhaustive on the need**:
    /// `AssetNeed::substrate()` answers `None` for a brush and for a picture alike, and
    /// a two-arm form files a picture in the brush store (§8, §23).
    pub fn install(&mut self, need: AssetNeed, bytes: &[u8]) -> stark_engine::Result<()> {
        match need {
            AssetNeed::Brush(expected) => {
                let actual = self.engine.import_brush(bytes)?;
                if actual != expected {
                    return Err(DocError::Misnamed {
                        expected: need,
                        actual: AssetNeed::Brush(actual),
                    }
                    .into());
                }
                Ok(())
            }
            AssetNeed::Substrate(id) => self
                .engine
                .accept_substrate(SubstrateId::Image(id), bytes)
                .map(drop),
            AssetNeed::Picture(id) => self.engine.accept_picture(id, bytes),
        }
    }

    /// Replace the document: install what it `owed`, replace, and frame the piece
    /// (§8, §12.4, §15.6).
    ///
    /// Refused, with the open document untouched, when the engine declines, or when an
    /// install fails for anything but a join; content installed before the refusal stays,
    /// as imported content does. A join's failed install is a promise this client
    /// cannot keep: it is skipped and answered in [`Replaced::skipped`], and the peer
    /// fetch supplies it (`stark_net::Joined::owed`).
    pub fn replace(
        &mut self,
        replacement: Replacement<'_>,
        owed: &[(AssetNeed, &[u8])],
    ) -> stark_engine::Result<Replaced> {
        let joining = matches!(replacement, Replacement::Join(..));
        let mut skipped = Vec::new();
        // Before the replay: a substrate not registered when its `SetSubstrate` replays
        // deposits every later stroke through the flat stand-in (§6.4).
        for &(need, bytes) in owed {
            match self.install(need, bytes) {
                Ok(()) => {}
                Err(error) if joining => skipped.push((need, error)),
                Err(error) => return Err(error),
            }
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "the one door every consumer's clippy.toml holds the rest to"
        )]
        match replacement {
            Replacement::Open(file) => self.engine.load_document(file)?,
            Replacement::Join(file, identity) => self.engine.join_collaboration(file, identity)?,
            Replacement::New(space, substrate) => self.engine.new_document(space, substrate)?,
        }
        // A view is per-client and in neither a file nor a snapshot (§18.1.2), so without
        // this a document arrives at the last one's pan and zoom.
        let mut seen = self.engine.observe();
        self.engine
            .process(ViewCommand::ShowPiece(crate::bounds::piece_frame(&seen)));
        // Framing moves the view and nothing else the projection holds, so the one built
        // before it is patched rather than a second one walked (`tests/desk.rs`).
        seen.view = self.engine.view();
        Ok(Replaced { seen, skipped })
    }
}

/// Shipped assets by catalog name, recorded as each is loaded.
///
/// An id is the hash of an asset's bytes, so a name answers only once they are in. A
/// document may be on content in no catalog, so this is a *display* index, never the
/// source of truth. A short list asked by name a handful of times a frame: a `Vec`
/// beats a map, and keeps catalog order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded<Id> {
    rows: Vec<(&'static str, Id)>,
}

impl<Id> Default for Loaded<Id> {
    fn default() -> Self {
        Self { rows: Vec::new() }
    }
}

impl<Id: Copy> Loaded<Id> {
    /// The id `name` was loaded under.
    pub fn get(&self, name: &str) -> Option<Id> {
        self.rows
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, id)| *id)
    }

    /// Record that `name` loaded as `id`. A repeat replaces the row rather than adding
    /// one.
    fn record(&mut self, name: &'static str, id: Id) {
        self.rows.retain(|(n, _)| *n != name);
        self.rows.push((name, id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(n: u8) -> SubstrateId {
        SubstrateId::Image(AssetId([n; 32]))
    }

    /// A name answers the id it was loaded under, and nothing before it was loaded.
    #[test]
    fn a_name_answers_only_once_loaded() {
        let mut shapes = Loaded::default();
        let id = AssetId([3; 32]);
        assert_eq!(shapes.get("Pencil"), None, "nothing is loaded yet");
        shapes.record("Pencil", id);
        assert_eq!(shapes.get("Pencil"), Some(id));
        assert_eq!(
            shapes.get("Flat"),
            None,
            "one load answers for its own name alone"
        );
    }

    /// Recording a name twice leaves the index as recording it once did — and a record
    /// under another id (a re-canonicalized one) replaces the row.
    #[test]
    fn recording_a_name_twice_keeps_one_row() {
        let mut once = Loaded::default();
        once.record("Linen", image(1));
        let mut twice = once.clone();
        twice.record("Linen", image(1));
        assert_eq!(twice, once, "a repeat record adds no second row");

        twice.record("Linen", image(2));
        assert_eq!(twice.get("Linen"), Some(image(2)));
        assert_eq!(twice.rows.len(), 1, "and a moved id replaces its row");
    }
}
