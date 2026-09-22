//! Every entry point the shaders declare, against the pipelines the engine builds
//! (§6.10). Nothing here changes the engine.
//!
//! **Both halves are discovered.** What the tree declares comes from
//! [`stark_shaders::every_build`], generated the way the rest of the mirror is; what the
//! engine builds is recorded as it builds it, through the one funnel every pipeline
//! crosses ([`desc::Module::declares`](super::desc)). So this file holds neither a list
//! of pipelines nor a list of shaders — it held both, ~400 lines of them, and a
//! hand-written list is exactly as good as whoever last edited it.
//!
//! What a list could not catch, and this does: a pipeline the engine stopped building,
//! and a shader added to the tree that nothing was ever written for.

use std::collections::BTreeSet;

use crate::gpu::desc::record::{Built, Recording};

/// One entry point an artifact declares that no pipeline builds, and why.
struct Unbuilt {
    artifact: &'static str,
    entry: &'static str,
    why: &'static str,
}

impl Unbuilt {
    fn at(&self) -> Built {
        (self.artifact, self.entry)
    }
}

/// The entry points compiled into an artifact and built into nothing.
///
/// The sweep is linked along the ceiling axis as well as the residual, and `fs_levels`
/// is not `@if(ceiling)`-gated — it writes the lane and nothing else, so the same
/// function is right in both builds and the gate would say nothing. The stamp loop's
/// per-segment draw of the lane runs the **plain** module's (`swept::SWEEPS`),
/// so the ceiling build translates a second copy that no pipeline names.
///
/// Costs a translation per ceiling variant and nothing else; gating it in the WESL
/// would retire these rows, which is what makes them worth stating.
const UNBUILT: &[Unbuilt] = &[
    Unbuilt {
        artifact: "stamp_ceiling",
        entry: "fs_levels",
        why: "not `@if(ceiling)`-gated, and the levels pipeline runs the plain build's",
    },
    Unbuilt {
        artifact: "stamp_resid_ceiling",
        entry: "fs_levels",
        why: "the same, in the pigment space",
    },
];

/// Every entry point this build deposited, keyed the way the recorder keys one.
fn declared() -> BTreeSet<Built> {
    stark_shaders::every_build()
        .iter()
        .flat_map(|(_, entries)| entries.iter().map(|ep| (ep.artifact, ep.name)))
        .collect()
}

/// Everything the engine builds a pipeline from, over every colour space this build
/// carries — recorded, not listed.
///
/// The union rather than a set per space: an [`EntryPoint`](stark_shaders::EntryPoint)
/// names its own artifact, so `media_oklab`'s `fs_main` and `media_mixbox`'s are
/// different members and nothing is lost by putting them in one set.
///
/// One engine per space is the whole of it: every pipeline this engine has is compiled
/// by the constructor, the compositor's included, so what is recorded is a fact about
/// the *build* rather than about a script of gestures. A pass that became lazy would
/// show up below as `unbuilt`, and the answer then is to provoke it here — not to
/// excuse it in [`UNBUILT`], which is for what is compiled and genuinely never run.
fn built(ctx: &crate::gpu::context::GpuContext) -> BTreeSet<Built> {
    let mut out = BTreeSet::new();
    for space in crate::colorspace::all_available() {
        let recording = Recording::open();
        // Built for what it compiles, and read for nothing: the pipelines are the
        // observation, and the recording below is where they land.
        let _engine = crate::Engine::new_with_color_space(
            ctx.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
            crate::view::Extent2::new(64, 64),
            space,
        )
        .expect("`all_available` names only spaces this build carries");
        out.extend(recording.take());
    }
    out
}

/// Every entry point the shaders declare is built into some pipeline, **except** the
/// declared few — and every pipeline the engine builds is one the shaders declare.
///
/// An **exact** set against [`UNBUILT`], so a row that has stopped excusing anything
/// fails as loudly as an entry point nobody triaged — which is how an allow-list turns
/// into the place the next mistake hides.
#[test]
fn every_entry_point_declared_is_built_into_some_pipeline() {
    let headless = pollster::block_on(crate::gpu::context::GpuContext::headless());
    let Some(ctx) = crate::testing::or_skip(headless, "GPU tests") else {
        return;
    };
    let (declared, built) = (declared(), built(&ctx));
    assert!(
        !declared.is_empty() && !built.is_empty(),
        "neither side may be empty, or this agrees with itself about nothing",
    );

    let here = |u: &&Unbuilt| declared.contains(&u.at());
    let excused: BTreeSet<Built> = UNBUILT.iter().filter(here).map(Unbuilt::at).collect();
    let say = |(artifact, entry): &Built| format!("`{artifact}`'s `{entry}`");

    let mut bad: Vec<String> = declared
        .difference(&built)
        .filter(|w| !excused.contains(*w))
        .map(|w| format!("unbuilt: {}", say(w)))
        .collect();
    bad.extend(
        UNBUILT
            .iter()
            .filter(here)
            .filter(|u| built.contains(&u.at()))
            .map(|u| format!("stale:   {} — declared as {}", say(&u.at()), u.why)),
    );
    // The other direction, which no hand-written table could answer: something was
    // built that this tree does not declare. Unreachable while every pipeline is made
    // from a generated record, and that is the claim.
    bad.extend(
        built
            .difference(&declared)
            .map(|w| format!("foreign: {}", say(w))),
    );
    assert!(
        bad.is_empty(),
        "an `unbuilt` line is an entry point this build compiles and no pipeline names \
         — the engine stopped building it, nobody ever did, or it is now built lazily \
         and `built` has to provoke it; a `stale` one is a row excusing nothing; a \
         `foreign` one is a pipeline built over an entry point no deposited artifact \
         declares:\n{}",
        bad.join("\n"),
    );
}
