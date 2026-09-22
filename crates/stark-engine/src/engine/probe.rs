//! [`Probe`], the suite's diagnostic reads (§9). Implemented inside `engine` so each
//! reads the state it reports directly; `Engine::x(self)` below names an inherent
//! method of the same name, which method resolution prefers over the trait's.

use stark_model::SubstrateId;
use stark_model::geom::Vec2;
#[cfg(not(target_arch = "wasm32"))]
use stark_model::{document::LayerId, geom::TileCoord};

use super::Engine;
#[cfg(not(target_arch = "wasm32"))]
use crate::RgbaImage;
use crate::document::DocState;
use crate::gpu::substrate::Substrate;
#[cfg(not(target_arch = "wasm32"))]
use crate::testing::TileChannels;
use crate::testing::{Probe, TimelineStats};

impl Probe for Engine {
    fn document(&self) -> &DocState {
        Engine::document(self)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn tile_channels(&self, layer: LayerId, coord: TileCoord) -> Option<TileChannels> {
        Some(
            Engine::document(self)
                .layer(layer)?
                .tiles()?
                .get(&coord)?
                .read_channels(&self.shared.gpu),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_to_image(&mut self) -> RgbaImage {
        Engine::render_to_image(self)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_to_floats(&mut self) -> Vec<f32> {
        Engine::render_to_floats(self)
    }

    fn flush_live(&mut self) {
        Engine::flush_live(self);
    }

    fn timeline_stats(&self) -> TimelineStats {
        self.timeline.stats()
    }

    fn live_head_count(&mut self) -> usize {
        Engine::flush_live(self);
        self.preview.head_count()
    }

    fn strokes_reused(&self) -> u64 {
        self.strokes_reused
    }

    fn preview_epoch(&self) -> u64 {
        self.preview.epoch()
    }

    fn preview_fold(&mut self) -> u64 {
        Engine::flush_live(self);
        self.preview.fold()
    }

    fn substrate_bearing(&self, id: SubstrateId, give: f32, softness: f32, dir: Vec2) -> f32 {
        let substrate = Substrate {
            id,
            // The document's scale: the substrate a stroke would bite right now.
            ..Engine::document(self).substrate()
        };
        self.shared
            .apply
            .substrates
            .get(&self.shared.gpu, substrate)
            .bearing(give, softness, dir.to_array())
    }

    fn resident_tile_bytes(&self) -> u64 {
        self.shared.apply.pool.resident_bytes()
    }

    fn take_scratch_peak(&mut self) -> u64 {
        self.shared.scratch.take_peak()
    }
}
