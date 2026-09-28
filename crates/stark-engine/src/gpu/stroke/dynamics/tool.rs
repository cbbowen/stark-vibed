//! The loop's **tool reservoir** (§6.2): the ping-pong pair a dispatch reads one half
//! of and writes the other, and the copies that carry it between ranges.
//!
//! What the tip carries is the whole of what the loop threads between segments which
//! is not already on the canvas, so a [`Reservoir`] is the same three channels this
//! pair holds. They are cut with the **same pool key** the halves are taken at, which
//! is what makes [`resume`](ToolPingPong::resume) and
//! [`capture`](ToolPingPong::capture) one procedure run in opposite directions.

use stark_shaders::Binding;
use stark_shaders::mirror::dynamics::decl as d;

use super::super::incremental::Reservoir;
use super::super::{StrokeConstants, StrokeRenderer};
use super::region::LOOP_USAGE;
use super::trio::{Life, Trio};
use crate::gpu::desc;
use crate::gpu::scratch::{Kept, Key, SubmitScope};

/// Resolution (texels per side) of the stamp loop's tool reservoir (§6.2).
/// Brush-local, so carried color detail is ~radius/32 canvas px — plenty for smeared
/// paint, and small enough that the per-stamp reservoir update is nearly free.
pub(super) const BRUSH_RES: u32 = 64;

/// The three declarations a half is storage-written through (§6.10) — the formats a
/// half, and every copy of one, is created at.
const WRITES: [Binding; 3] = [
    d::BRUSH_DST_COLOR_W,
    d::BRUSH_DST_AUX_W,
    d::BRUSH_DST_RESID_W,
];

/// What a reservoir texture has to be. A stroke that starts fresh initializes its
/// first half by a render clear (the driver does the f16 encode), hence
/// `RENDER_ATTACHMENT`; one resuming copies into it instead, hence the `COPY` pair —
/// which also carries the end state back out.
const USAGE: wgpu::TextureUsages = LOOP_USAGE
    .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// The extent of every copy between a half and a carried reservoir.
const EXTENT: wgpu::Extent3d = wgpu::Extent3d {
    width: BRUSH_RES,
    height: BRUSH_RES,
    depth_or_array_layers: 1,
};

/// Channel `i`'s pool key, at the format its own declaration names (§6.10).
///
/// **Both ends of the carry ask this one function**, so a captured texture and the
/// half it was cut from land on one line of the pool's free list rather than on two
/// that happen to hold the same picture. The label is out of the match ([`Key`]).
fn key(i: usize, label: &'static str) -> Key {
    Key {
        size: (BRUSH_RES, BRUSH_RES),
        format: WRITES[i].storage_format(),
        usage: USAGE,
        label,
    }
}

/// The tool reservoir as the loop holds it (§6.2): one whole channel trio per half —
/// the residual among them (§6.7), so the tip carries the rest of the color it picked
/// up rather than the grey a pigment space's concentrations alone describe — and which
/// half holds the tool now.
pub(super) struct ToolPingPong {
    halves: [Trio; 2],
    cur: usize,
}

impl ToolPingPong {
    /// Check the pair out of the pool, one whole trio a half. Fully written — cleared,
    /// copied into, or stored texel-for-texel by the reservoir passes — before anything
    /// reads it, which is the pool's no-zero-init contract (`scratch`).
    pub(super) fn take(scope: &mut SubmitScope, resid: bool) -> Self {
        let half = |scope: &mut SubmitScope, labels: [&'static str; 3]| {
            let keys = std::array::from_fn(|i: usize| key(i, labels[i]));
            Trio::take_at(scope, Life::Run, keys, resid)
        };
        Self {
            halves: [
                half(
                    scope,
                    [
                        "stark dynamics brush color a",
                        "stark dynamics brush aux a",
                        "stark dynamics brush resid a",
                    ],
                ),
                half(
                    scope,
                    [
                        "stark dynamics brush color b",
                        "stark dynamics brush aux b",
                        "stark dynamics brush resid b",
                    ],
                ),
            ],
            cur: 0,
        }
    }

    /// Put the tool in the state `from` left it in: the tip arrives at this range
    /// carrying exactly what it carried when the previous range stopped.
    pub(super) fn resume(&self, scope: &mut SubmitScope, from: &Reservoir) {
        let into = self.src(self.cur);
        let mut copy = |src: &Kept, dst: &wgpu::Texture| {
            scope.encoder().copy_texture_to_texture(
                src.tex().as_image_copy(),
                dst.as_image_copy(),
                EXTENT,
            );
        };
        copy(&from.color, &into.color.tex);
        copy(&from.aux, &into.aux.tex);
        if let (Some(src), Some(dst)) = (&from.resid, into.resid_tex()) {
            copy(src, dst);
        }
    }

    /// Charge a fresh tip: latent = the brush's own color, per-unit opacity exactly 1
    /// (minted paint is opaque per unit, §6.2), and the carried amount at the
    /// pre-`charge` glob (0 = empty tool).
    pub(super) fn charge(&self, scope: &mut SubmitScope, k: &StrokeConstants, charge: f32) {
        let into = self.src(self.cur);
        desc::begin_pass(
            scope.encoder(),
            "stark dynamics brush init",
            // The three clears say unlike things, so each attachment carries its own
            // (`desc::Attachments::at_locations`) and the residual's absence is a hole
            // rather than a count to slice to.
            desc::Attachments::at_locations([
                Some(desc::attach(
                    &into.color.view,
                    desc::clear_to(wgpu::Color {
                        r: k.channels[0] as f64,
                        g: k.channels[1] as f64,
                        b: k.channels[2] as f64,
                        a: 1.0,
                    }),
                )),
                Some(desc::attach(
                    &into.aux.view,
                    desc::clear_to(wgpu::Color {
                        // Carried height is the only thing the tool carries (§6.1),
                        // so the rest of the reservoir aux is unused. Opacity scales
                        // the glob outright, and for a finite source that *is* the
                        // ceiling — so no budget lane is needed here, where the
                        // `add` rate's unbounded mint takes one (§6.2,
                        // `dynamics.wesl::lay_parcel`).
                        r: (charge * k.opacity) as f64,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                )),
                // A freshly charged tip holds the brush's own color, and that
                // color's residual with it — the same clear, off the same constants
                // (§6.7).
                into.resid_view().map(|v| {
                    desc::attach(
                        v,
                        desc::clear_to(wgpu::Color {
                            r: k.resid[0] as f64,
                            g: k.resid[1] as f64,
                            b: k.resid[2] as f64,
                            a: 0.0,
                        }),
                    )
                }),
            ]),
        );
    }

    /// Cut the tool out for the range that resumes after this one. Copied rather than
    /// aliased: the halves go back to the pool when the run's scope closes, and the
    /// range that resumes writes its first exchange straight into whatever it is
    /// handed. 64² rgba16f ×2, so ~64 KB.
    ///
    /// The copies are pooled [`Kept`] leases — one is captured per pointer move, and
    /// the drop that returns one is provably behind the resuming run's submit.
    pub(super) fn capture(&self, r: &StrokeRenderer, scope: &mut SubmitScope) -> Reservoir {
        let held = self.src(self.cur);
        let mut cut = |i: usize, src: &wgpu::Texture, label| {
            let dst = r.scratch.keep(&r.ctx.device, key(i, label));
            scope.encoder().copy_texture_to_texture(
                src.as_image_copy(),
                dst.tex().as_image_copy(),
                EXTENT,
            );
            dst
        };
        Reservoir {
            color: cut(0, &held.color.tex, "stark tool state color"),
            aux: cut(1, &held.aux.tex, "stark tool state aux"),
            // Carried across ranges with the color, and for the same reason: a tip that
            // lifted black paint has to still be carrying black when the next pointer
            // move resumes it (§6.7).
            resid: held
                .resid_tex()
                .map(|t| cut(2, t, "stark tool state resid")),
        }
    }

    /// The half a dispatch at `phase` reads…
    pub(super) fn src(&self, phase: usize) -> &Trio {
        &self.halves[phase]
    }

    /// …and the half it writes. The residual turns on the same phase as the color, or
    /// the tool's two halves would drift apart.
    pub(super) fn dst(&self, phase: usize) -> &Trio {
        &self.halves[1 - phase]
    }

    /// Which half holds the tool: what this segment's bake reads, and what a capture
    /// cuts from.
    pub(super) fn cur(&self) -> usize {
        self.cur
    }

    /// Hand the tool to the other half — where the segment's exchange just wrote a tip
    /// that has travelled and reloaded.
    pub(super) fn advance(&mut self) {
        self.cur = 1 - self.cur;
    }
}
