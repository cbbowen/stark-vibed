//! Every hand-written slot list, against what the entry points bound through it
//! actually read (§6.10).
//!
//! A `&[Slot]` list is a **membership claim**: these are the bindings, and these are the
//! ones read through a sampler. Both halves are things the linked shader knows — naga
//! works them out while type-checking, callees included, and the generated
//! [`EntryPoint::uses`] carries the answer — and until now nothing compared the two. A
//! list naming one slot too many is a layout entry and a bind-group entry no shader
//! reads, which `wgpu` accepts in silence; one too few is a pipeline that will not
//! create, on a GPU, in whichever colour space links that variant.
//!
//! **The unit is the list, not the pipeline.** Several pipelines share one layout — the
//! blur's four kernels, the filter's three fragment entry points, the sweep's five —
//! and a shared layout is necessarily the *union* of what its pipelines read, so a
//! per-pipeline comparison would report every sharer's unused entries as a fault. The
//! table below is still the pipelines, because that is what a reader can check against
//! `desc`'s call sites; the check folds them by list.
//!
//! `PREFIX_SLOTS` is the other shape and the union is weaker there: one list becomes
//! **two** layout objects, the sweep's fragment-visible one and the wet loop's
//! compute-visible one, so what is checked is the union over both.
//!
//! **It is not a `#[derive]` for the lists.** What a list still says, and the shader
//! cannot, is which `@group` a pipeline binds it as, in what order, and at what
//! visibility. The list stays written by hand; this says whether it is true.
//!
//! [`KNOWN`] is the differences that stand today, each with why — and the check is that
//! the differences found **equal** it, so a waiver cannot outlive what it excuses.
//! Nothing here changes the engine.

use std::collections::{BTreeMap, BTreeSet};

use stark_shaders::{EntryPoint, Lane, Resid};

use crate::gpu::composite::{blend, blur, filter, guides, media, overlay, resolve, tiles};
use crate::gpu::desc::Slot;
use crate::gpu::stroke::dynamics::{kit, slots as dyn_slots};
use crate::gpu::stroke::{erase, swept};
use crate::gpu::{fill, merge, selection, transform};

/// One bind group layout a pipeline binds: the list it is built from, named as the
/// host writes it, and **the residual the host built it under** — which is not always
/// the colour space's. The sweep, the prefix tap and the overlay pass `false`
/// unconditionally, having no `@if(resid)` declaration between them.
#[derive(Clone, Copy)]
struct Group {
    name: &'static str,
    slots: &'static [Slot],
    resid: bool,
}

const fn group(name: &'static str, slots: &'static [Slot], resid: bool) -> Group {
    Group { name, slots, resid }
}

/// One pipeline the engine creates: the entry points it is built from, and the groups
/// its pipeline layout lists.
struct Case {
    /// The label `desc` gives `wgpu`, shortened.
    what: &'static str,
    /// Every stage's entry point. A render pipeline's layout has to satisfy the vertex
    /// and fragment stages together, so the two are read as one set.
    entries: Vec<EntryPoint>,
    groups: Vec<Group>,
}

/// A slot as both sides can name it: the module that declares it, and the declaration.
///
/// Neither alone is unique — three modules partition group 0 between them in the blend
/// and filter layouts, and `stamp_common`'s `xf` is not `transform`'s.
type Key = (&'static str, &'static str);

/// Which way a list and its shader disagree.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Diff {
    /// The list names a slot no entry point bound through it reads.
    Unread,
    /// The list and the shader disagree about whether it is read through a sampler.
    Sampling,
    /// The list lacks a slot an entry point bound through it reads — the pipeline
    /// cannot be created. Never waived: there is nothing to trade against a layout the
    /// device refuses.
    Omitted,
}

/// One difference, as both sides can name it — and the key the found set and the
/// declared one are compared on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Where {
    /// Whether it stands in the space that carries a residual (§6.7). **Part of the
    /// key**, because a difference is a fact about one space: the pigment LUT is a
    /// placeholder in Oklab and the real table in Mixbox, so a waiver keyed without
    /// this would cover a genuine fault in the other.
    resid: bool,
    list: &'static str,
    module: &'static str,
    decl: &'static str,
    diff: Diff,
}

/// One difference that stands today, and why it is not a fault.
///
/// Flat, and its [`Where`] built rather than nested, so no row can pair a list with
/// another row's declaration.
struct Known {
    resid: bool,
    list: &'static str,
    module: &'static str,
    decl: &'static str,
    diff: Diff,
    why: &'static str,
}

impl Known {
    const fn at(&self) -> Where {
        Where {
            resid: self.resid,
            list: self.list,
            module: self.module,
            decl: self.decl,
            diff: self.diff,
        }
    }
}

/// The differences that stand today, one row per colour space each stands in.
///
/// **Two rows where one difference stands in both spaces**, because the space is part
/// of what is being declared: three of the seven below hold in Oklab and Mixbox alike,
/// and four are the colorimetric space's alone. None of them is fixed here — the lists
/// are another commit's business, and a check that quietly edits what it measures is
/// not a check.
const KNOWN: &[Known] = &[
    // One layout serves both colour spaces, because whether the pigment LUT is real is
    // `ColorSpace::needs_pigment_lut`'s answer rather than the layout's: the
    // colorimetric shaders do not import `mixbox_lut` at all, and the host binds a 1×1
    // stand-in — the same "one shader, one layout" the zero masks buy elsewhere (§6.8).
    // The pigment space reads the real table, so these four stand at `resid: false` and
    // nowhere else.
    Known {
        resid: false,
        list: "BLEND_SLOTS",
        module: "mixbox_lut",
        decl: "PIGMENT_LUT",
        diff: Diff::Unread,
        why: "a placeholder in the colorimetric space, which declares no LUT",
    },
    Known {
        resid: false,
        list: "BLEND_SLOTS",
        module: "mixbox_lut",
        decl: "PIGMENT_SAMP",
        diff: Diff::Unread,
        why: "the placeholder LUT's sampler",
    },
    Known {
        resid: false,
        list: "FILTER_SLOTS",
        module: "mixbox_lut",
        decl: "PIGMENT_LUT",
        diff: Diff::Unread,
        why: "a placeholder in the colorimetric space, which declares no LUT",
    },
    Known {
        resid: false,
        list: "FILTER_SLOTS",
        module: "mixbox_lut",
        decl: "PIGMENT_SAMP",
        diff: Diff::Unread,
        why: "the placeholder LUT's sampler",
    },
    // The three below are the lists being wrong rather than a layout being shared.
    // Left as they are and recorded here, since fixing a list changes a layout and a
    // bind group, which is a pixel-affecting change.
    //
    // The first two are unconditional declarations, so they stand in both spaces.
    Known {
        resid: false,
        list: "dynamics::DEPOSIT",
        module: "dynamics",
        decl: "SAMP",
        diff: Diff::Unread,
        why: "the bilinear sampler is `exchange`'s and `bake`'s; the deposit reads its \
              noise through `dyn_noise_samp` and everything else with `textureLoad`",
    },
    Known {
        resid: true,
        list: "dynamics::DEPOSIT",
        module: "dynamics",
        decl: "SAMP",
        diff: Diff::Unread,
        why: "the same, in the pigment space",
    },
    Known {
        resid: false,
        list: "dynamics::SETTLE",
        module: "dynamics",
        decl: "REGION_LEVELS",
        diff: Diff::Unread,
        why: "the list says the settle lays through `lay_parcel`, which reads the lane \
              — it does not: the pen-up builds its parcel in `settle` itself and stores \
              through `stack_and_store`",
    },
    Known {
        resid: true,
        list: "dynamics::SETTLE",
        module: "dynamics",
        decl: "REGION_LEVELS",
        diff: Diff::Unread,
        why: "the same, in the pigment space",
    },
    // And the third is `@if(resid)`, so it is in no list at all without the residual —
    // the pigment space alone. The allow-list this replaced was keyed without the space
    // and declared it at `resid: false`, where it never fired.
    Known {
        resid: true,
        list: "dynamics::EXCHANGE",
        module: "dynamics",
        decl: "BRUSH_SRC_RESID",
        diff: Diff::Sampling,
        why: "listed `sampled` beside its `at` partners `BRUSH_SRC_COLOR`/`_AUX`, which \
              `exchange` loads exactly as it loads the residual — the filterable flag \
              is `bake`'s, where all three really are sampled",
    },
];

/// What the shader says: every binding these entry points reach **in `@group(group)`**,
/// and whether any of them samples it.
///
/// Restricted to the one group, because a bind group layout describes exactly one — so
/// what a list is answerable for is its group's share of what the entry point reads,
/// and nothing else the pipeline binds beside it.
///
/// `|`, not the last one's answer: a layout entry is one entry for every stage and
/// every sharer, so a texture one of them samples has to be declared filterable even
/// where the rest load it.
fn shader_uses(entries: &[EntryPoint], group: u32) -> BTreeMap<Key, bool> {
    let mut out = BTreeMap::new();
    for ep in entries {
        for used in ep.uses.iter().filter(|u| u.decl.group == group) {
            *out.entry((used.decl.module, used.decl.name))
                .or_insert(false) |= used.sampled;
        }
    }
    out
}

/// What one list claims, with the residual gate it is built under applied.
fn host_list(g: Group) -> BTreeMap<Key, bool> {
    let mut out = BTreeMap::new();
    for slot in g.slots.iter().filter(|s| s.present(g.resid)) {
        *out.entry((slot.decl().module, slot.decl().name))
            .or_insert(false) |= slot.is_sampled();
    }
    out
}

/// The `@group` a list describes. `desc::layout_for` refuses a list spanning two, and
/// `slots.rs` states it for the wet loop's eleven without a device.
fn list_group(g: Group) -> u32 {
    g.slots
        .first()
        .expect("a slot list names at least one binding")
        .decl()
        .group
}

/// One shader record a pipeline here is built from: what the engine calls it, and
/// every entry point it declares.
///
/// The half of the table that makes the pipeline list *checkable*: a record's entry
/// points are the shaders' own answer, so an entry point no [`Case`] names is one the
/// engine declares and never builds.
struct Record {
    what: &'static str,
    entries: &'static [EntryPoint],
}

/// Every pipeline the engine creates in one colour space, and every record they are
/// built from.
struct Table {
    cases: Vec<Case>,
    records: Vec<Record>,
}

/// The colour-space-specific passes: a document runs the pigment media, blend and
/// filter shaders or the colorimetric ones, never both.
///
/// Shaped like [`ColorSpace::resid`](crate::colorspace::ColorSpace::resid) for the same
/// reason — without the `mixbox` feature the pigment half does not exist to be named.
fn space_table(resid: bool) -> Table {
    #[cfg(feature = "mixbox")]
    if resid {
        let (m, b, f) = (
            stark_shaders::media_mixbox(),
            stark_shaders::blend_mixbox(),
            stark_shaders::filter_mixbox(),
        );
        return space_shaped(
            resid,
            ("media_mixbox", m.entries, [m.vs_main, m.fs_main]),
            ("blend_mixbox", b.entries, [b.vs_main, b.fs_main]),
            (
                "filter_mixbox",
                f.entries,
                [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
            ),
        );
    }
    let (m, b, f) = (
        stark_shaders::media_oklab(),
        stark_shaders::blend_oklab(),
        stark_shaders::filter_oklab(),
    );
    space_shaped(
        resid,
        ("media_oklab", m.entries, [m.vs_main, m.fs_main]),
        ("blend_oklab", b.entries, [b.vs_main, b.fs_main]),
        (
            "filter_oklab",
            f.entries,
            [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
        ),
    )
}

/// One space's record, as [`space_shaped`] takes it: its name, everything it declares,
/// and the entry points the pipelines below name.
type Shader<const N: usize> = (&'static str, &'static [EntryPoint], [EntryPoint; N]);

/// The five pipelines the two spaces have the same shape of — one media pass, one
/// blend, and the filter's three fragment entry points over one layout.
fn space_shaped(resid: bool, media: Shader<2>, blend: Shader<2>, filter: Shader<4>) -> Table {
    let [vs, fs_main, fs_tile, fs_blur_decode] = filter.2;
    let one = |what, fs| Case {
        what,
        entries: vec![vs, fs],
        groups: vec![group("FILTER_SLOTS", filter::FILTER_SLOTS, resid)],
    };
    Table {
        cases: vec![
            Case {
                what: "media",
                entries: media.2.to_vec(),
                groups: vec![group("MEDIA_SLOTS", media::MEDIA_SLOTS, resid)],
            },
            Case {
                what: "blend",
                entries: blend.2.to_vec(),
                groups: vec![group("BLEND_SLOTS", blend::BLEND_SLOTS, resid)],
            },
            one("filter", fs_main),
            one("filter tile", fs_tile),
            one("filter blur decode", fs_blur_decode),
        ],
        records: vec![
            Record {
                what: media.0,
                entries: media.1,
            },
            Record {
                what: blend.0,
                entries: blend.1,
            },
            Record {
                what: filter.0,
                entries: filter.1,
            },
        ],
    }
}

/// Every pipeline the engine creates, for the colour space `r`.
#[expect(
    clippy::too_many_lines,
    reason = "one entry per pipeline, which is the point"
)]
fn table(r: Resid) -> Table {
    let resid = r.on();
    let composite = stark_shaders::composite(r);
    let matte = stark_shaders::matte(r);
    let ov = stark_shaders::overlay();
    let gu = stark_shaders::guides();
    let re = stark_shaders::resolve();
    let bl = stark_shaders::blur();
    let fi = stark_shaders::fill(r);
    let me = stark_shaders::merge(r);
    let sl = stark_shaders::slab(r);
    let tr = stark_shaders::transform(r);
    let se = stark_shaders::selection();
    let mr = stark_shaders::mask_region();
    let plain = stark_shaders::stamp(r, Lane::Plain);
    let ceiling = stark_shaders::stamp(r, Lane::Ceiling);
    let ig = stark_shaders::integrate(r);
    let er = stark_shaders::erase(r);
    let dy = stark_shaders::dynamics(r);
    let li = stark_shaders::liquify(r);
    let sc = stark_shaders::slice();

    // The composite's two groups, which the wet loop's own preview pass rebuilds from
    // the same two lists.
    let view = || group("composite::VIEW_SLOTS", tiles::VIEW_SLOTS, resid);
    let tile = || group("composite::TILE_SLOTS", tiles::TILE_SLOTS, resid);
    // The sweep's three, shared by five pipelines and built `false` throughout: nothing
    // `stamp_common` declares is `@if(resid)`.
    let sweep = || {
        vec![
            group("XFORM_SLOTS", swept::XFORM_SLOTS, false),
            group("PREFIX_SLOTS", swept::PREFIX_SLOTS, false),
            group("NOISE_SLOTS", swept::NOISE_SLOTS, false),
        ]
    };
    // The prefix tap the wet loop's kernels take at group 1, likewise — a second
    // layout object over the one list, compute-visible where the sweep's is the
    // fragment stage's.
    let prefix = || group("PREFIX_SLOTS", swept::PREFIX_SLOTS, false);
    let dyn_group = |name, slots| group(name, slots, resid);

    let mut all = vec![
        Case {
            what: "composite",
            entries: vec![composite.vs_main, composite.fs_main],
            groups: vec![view(), tile()],
        },
        Case {
            what: "matte",
            entries: vec![matte.vs_main, matte.fs_main],
            groups: vec![view(), group("RAMP_SLOTS", tiles::RAMP_SLOTS, resid)],
        },
        Case {
            what: "overlay",
            entries: vec![ov.vs_main, ov.fs_main],
            groups: vec![
                group("overlay::VIEW_SLOTS", overlay::VIEW_SLOTS, false),
                group("MASK_SLOTS", overlay::MASK_SLOTS, false),
            ],
        },
        Case {
            what: "guides",
            entries: vec![gu.vs_main, gu.fs_main],
            groups: vec![group("GUIDE_SLOTS", guides::GUIDE_SLOTS, false)],
        },
        Case {
            what: "resolve",
            entries: vec![re.vs_main, re.fs_main],
            groups: vec![group("RESOLVE_SLOTS", resolve::RESOLVE_SLOTS, false)],
        },
        Case {
            what: "blur fft",
            entries: vec![bl.fft_both],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur kernel fft",
            entries: vec![bl.fft_one],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur make kernel",
            entries: vec![bl.make_kernel],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "blur apply kernel",
            entries: vec![bl.apply_kernel],
            groups: vec![group("BLUR_SLOTS", blur::BLUR_SLOTS, false)],
        },
        Case {
            what: "fill",
            entries: vec![fi.vs_main, fi.fs_main],
            groups: vec![group("FILL_SLOTS", fill::FILL_SLOTS, resid)],
        },
        Case {
            what: "merge",
            entries: vec![me.vs_main, me.fs_main],
            groups: vec![group("MERGE_SLOTS", merge::MERGE_SLOTS, resid)],
        },
        Case {
            what: "slab expand",
            entries: vec![sl.vs_main, sl.fs_expand],
            groups: vec![group("SLAB_SLOTS", merge::SLAB_SLOTS, resid)],
        },
        Case {
            what: "slab store",
            entries: vec![sl.vs_main, sl.fs_store],
            groups: vec![group("SLAB_SLOTS", merge::SLAB_SLOTS, resid)],
        },
        Case {
            what: "transform parcel",
            entries: vec![tr.vs_quad, tr.fs_parcel],
            groups: vec![
                group("QUAD_SLOTS", transform::QUAD_SLOTS, resid),
                group("SRC_SLOTS", transform::SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform mask",
            entries: vec![tr.vs_quad, tr.fs_mask],
            groups: vec![
                group("QUAD_SLOTS", transform::QUAD_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform parcel gated",
            entries: vec![tr.vs_gated, tr.fs_parcel_gated],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("SRC_SLOTS", transform::SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform mask gated",
            entries: vec![tr.vs_gated, tr.fs_mask_gated],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "transform combine",
            entries: vec![tr.vs_fill, tr.fs_combine],
            groups: vec![group("COMBINE_SLOTS", transform::COMBINE_SLOTS, resid)],
        },
        Case {
            what: "transform mask base",
            entries: vec![tr.vs_fill, tr.fs_mask_base],
            groups: vec![
                group("GATED_SLOTS", transform::GATED_SLOTS, resid),
                group("MASK_SRC_SLOTS", transform::MASK_SRC_SLOTS, resid),
            ],
        },
        Case {
            what: "selection",
            entries: vec![se.vs_main, se.fs_main],
            groups: vec![group("RASTERIZE_SLOTS", selection::RASTERIZE_SLOTS, false)],
        },
        Case {
            what: "selection region",
            entries: vec![mr.vs_main, mr.fs_main],
            groups: vec![
                group("REGION_VIEW_SLOTS", selection::REGION_VIEW_SLOTS, false),
                group("REGION_TILE_SLOTS", selection::REGION_TILE_SLOTS, false),
            ],
        },
        Case {
            what: "sweep",
            entries: vec![plain.vs_main, plain.fs_main],
            groups: sweep(),
        },
        Case {
            what: "sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_main],
            groups: sweep(),
        },
        Case {
            what: "sweep levels",
            entries: vec![plain.vs_main, plain.fs_levels],
            groups: sweep(),
        },
        Case {
            what: "erase sweep",
            entries: vec![plain.vs_main, plain.fs_erase],
            groups: sweep(),
        },
        Case {
            what: "erase sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_erase],
            groups: sweep(),
        },
        Case {
            what: "integrate",
            entries: vec![ig.vs_main, ig.fs_main],
            groups: vec![group("INTEGRATE_SLOTS", swept::INTEGRATE_SLOTS, resid)],
        },
        Case {
            what: "erase",
            entries: vec![er.vs_main, er.fs_main],
            groups: vec![group("ERASE_SLOTS", erase::ERASE_SLOTS, resid)],
        },
        Case {
            what: "dynamics composite",
            entries: vec![composite.vs_main, composite.fs_raw],
            groups: vec![view(), tile()],
        },
        Case {
            what: "dynamics snapshot",
            entries: vec![dy.snapshot],
            groups: vec![dyn_group("dynamics::SNAPSHOT", dyn_slots::SNAPSHOT)],
        },
        Case {
            what: "dynamics bleed weight",
            entries: vec![dy.bleed_weight],
            groups: vec![
                dyn_group("dynamics::BLEED_WEIGHT", dyn_slots::BLEED_WEIGHT),
                prefix(),
            ],
        },
        Case {
            what: "dynamics exchange",
            entries: vec![dy.exchange],
            groups: vec![dyn_group("dynamics::EXCHANGE", dyn_slots::EXCHANGE)],
        },
        Case {
            what: "dynamics bake",
            entries: vec![dy.bake],
            groups: vec![dyn_group("dynamics::BAKE", dyn_slots::BAKE), prefix()],
        },
        Case {
            what: "dynamics deposit",
            entries: vec![dy.deposit],
            groups: vec![dyn_group("dynamics::DEPOSIT", dyn_slots::DEPOSIT), prefix()],
        },
        Case {
            what: "dynamics cell hoist",
            entries: vec![dy.cell_hoist],
            groups: vec![dyn_group("dynamics::HOIST", dyn_slots::HOIST), prefix()],
        },
        Case {
            what: "dynamics deposit coarse",
            entries: vec![dy.deposit_coarse],
            groups: vec![dyn_group(
                "dynamics::DEPOSIT_COARSE",
                dyn_slots::DEPOSIT_COARSE,
            )],
        },
        Case {
            what: "dynamics settle",
            entries: vec![dy.settle],
            groups: vec![dyn_group("dynamics::SETTLE", dyn_slots::SETTLE), prefix()],
        },
        Case {
            what: "liquify snapshot field",
            entries: vec![li.snapshot_field],
            groups: vec![dyn_group(
                "liquify::SNAPSHOT_FIELD",
                dyn_slots::SNAPSHOT_FIELD,
            )],
        },
        Case {
            what: "liquify warp",
            entries: vec![li.warp],
            groups: vec![dyn_group("liquify::WARP", dyn_slots::WARP), prefix()],
        },
        Case {
            what: "liquify warp apply",
            entries: vec![li.warp_apply],
            groups: vec![dyn_group("liquify::WARP_APPLY", dyn_slots::WARP_APPLY)],
        },
        Case {
            what: "dynamics slice",
            entries: vec![sc.vs_main, sc.fs_main],
            groups: vec![group("SLICE_SLOTS", kit::SLICE_SLOTS, false)],
        },
    ];
    // One line per accessor, from the very values the cases above are built out of —
    // so the set an entry point has to appear in is the shaders' own.
    let mut records = vec![
        Record {
            what: "composite",
            entries: composite.entries,
        },
        Record {
            what: "matte",
            entries: matte.entries,
        },
        Record {
            what: "overlay",
            entries: ov.entries,
        },
        Record {
            what: "guides",
            entries: gu.entries,
        },
        Record {
            what: "resolve",
            entries: re.entries,
        },
        Record {
            what: "blur",
            entries: bl.entries,
        },
        Record {
            what: "fill",
            entries: fi.entries,
        },
        Record {
            what: "merge",
            entries: me.entries,
        },
        Record {
            what: "slab",
            entries: sl.entries,
        },
        Record {
            what: "transform",
            entries: tr.entries,
        },
        Record {
            what: "selection",
            entries: se.entries,
        },
        Record {
            what: "mask_region",
            entries: mr.entries,
        },
        // The plain build alone: the generator checks that every variant of a shader
        // declares the same entry points, so the ceiling build adds no name.
        Record {
            what: "stamp",
            entries: plain.entries,
        },
        Record {
            what: "integrate",
            entries: ig.entries,
        },
        Record {
            what: "erase",
            entries: er.entries,
        },
        Record {
            what: "dynamics",
            entries: dy.entries,
        },
        Record {
            what: "liquify",
            entries: li.entries,
        },
        Record {
            what: "slice",
            entries: sc.entries,
        },
    ];
    let space = space_table(resid);
    all.extend(space.cases);
    records.extend(space.records);
    Table {
        cases: all,
        records,
    }
}

/// The table folded by list: each one, and every entry point bound through it.
///
/// # Panics
/// On a case that claims nothing — no entry point, or no group — on a group vector
/// whose order is not the `@group` numbering the shader declares, or if one name
/// stands for two lists or for one list at two residuals. Each would make the union
/// below a comparison against something no pipeline is.
fn by_list(cases: &[Case]) -> BTreeMap<&'static str, (Group, Vec<EntryPoint>)> {
    let mut out: BTreeMap<&'static str, (Group, Vec<EntryPoint>)> = BTreeMap::new();
    for case in cases {
        assert!(
            !case.entries.is_empty(),
            "`{}` names no entry point, so it says nothing about any list it holds",
            case.what,
        );
        assert!(
            !case.groups.is_empty(),
            "`{}` names no group, so nothing of it is checked",
            case.what,
        );
        for (i, g) in case.groups.iter().enumerate() {
            // A pipeline layout is positional: the i-th layout *is* `@group(i)`. The
            // declaration says which group a list is, so the two are compared here —
            // a swapped pair is otherwise a device-only failure.
            assert_eq!(
                list_group(*g),
                i as u32,
                "`{}` binds `{}` at position {i} of its pipeline layout, where the \
                 shader declares that list `@group({})`",
                case.what,
                g.name,
                list_group(*g),
            );
        }
        for g in &case.groups {
            let (seen, entries) = out.entry(g.name).or_insert_with(|| (*g, Vec::new()));
            // By content, not by address: a `const` reference is re-evaluated at every
            // use, so two mentions of one list need not be one allocation.
            let same = seen.resid == g.resid
                && seen.slots.len() == g.slots.len()
                && seen
                    .slots
                    .iter()
                    .zip(g.slots)
                    .all(|(a, b)| a.decl() == b.decl() && a.is_sampled() == b.is_sampled());
            assert!(
                same,
                "`{}` names two different lists, or one list at two residuals — the \
                 second is `{}`'s",
                g.name, case.what,
            );
            entries.extend(case.entries.iter().copied());
        }
    }
    out
}

/// Every way one list and what is bound through it disagree, in the space `resid`.
///
/// Unfiltered: the waivers are applied by comparing this whole set against [`KNOWN`],
/// not by excusing rows one at a time as they are found.
fn differences(resid: bool, list: &'static str, g: Group, entries: &[EntryPoint]) -> Vec<Where> {
    let used = shader_uses(entries, list_group(g));
    let listed = host_list(g);
    let mut out = Vec::new();
    let mut at = |key: Key, diff| {
        out.push(Where {
            resid,
            list,
            module: key.0,
            decl: key.1,
            diff,
        });
    };
    for (key, sampled) in &listed {
        match used.get(key) {
            None => at(*key, Diff::Unread),
            Some(is) if is != sampled => at(*key, Diff::Sampling),
            Some(_) => {}
        }
    }
    for key in used.keys() {
        if !listed.contains_key(key) {
            at(*key, Diff::Omitted);
        }
    }
    out
}

/// One difference in words.
fn describe(w: &Where) -> String {
    let (verb, tail) = match w.diff {
        Diff::Unread => ("lists", "which nothing bound through it reads"),
        Diff::Sampling => (
            "lists",
            "which it and the shader disagree about reading through a sampler",
        ),
        Diff::Omitted => (
            "omits",
            "which is bound through it — the pipeline cannot be created",
        ),
    };
    format!(
        "resid={}: `{}` {verb} `{}.wesl`'s `{}`, {tail}",
        w.resid, w.list, w.module, w.decl,
    )
}

/// The colour spaces this build has, which every check here runs over.
fn spaces() -> Vec<Resid> {
    #[cfg(feature = "mixbox")]
    let spaces = vec![Resid::Without, Resid::With];
    #[cfg(not(feature = "mixbox"))]
    let spaces = vec![Resid::Without];
    spaces
}

/// Every list against what is bound through it, in every colour space this build has.
fn found() -> BTreeSet<Where> {
    let mut out = BTreeSet::new();
    for r in spaces() {
        let table = table(r);
        for (list, (g, entries)) in by_list(&table.cases) {
            out.extend(differences(r.on(), list, g, &entries));
        }
    }
    out
}

/// The hand-written slot lists differ from the shaders in **exactly** the declared
/// places (§6.10).
///
/// An exact set rather than an allow-list, so the two ways of being wrong fail the
/// same way: a difference nobody triaged, and a waiver that has stopped excusing
/// anything. The second is how an allow-list turns into a place the next mistake
/// hides — the day `blend_oklab` declares a LUT of its own, those two rows should
/// fail rather than quietly cover something else.
///
/// The declared set is narrowed to the spaces this build links: a row for the pigment
/// space is unexercised, not stale, in a build without it.
///
/// Needs no device: the declarations are `const`s and the uses are generated.
#[test]
fn the_slot_lists_differ_from_the_shaders_exactly_where_declared() {
    let found = found();
    let here = |w: &Where| spaces().iter().any(|r| r.on() == w.resid);
    let declared: BTreeSet<Where> = KNOWN.iter().map(Known::at).filter(here).collect();
    let mut bad: Vec<String> = found
        .difference(&declared)
        .map(|w| format!("new:   {}", describe(w)))
        .collect();
    bad.extend(
        KNOWN
            .iter()
            .filter(|k| here(&k.at()) && !found.contains(&k.at()))
            .map(|k| format!("stale: {} — declared as {}", describe(&k.at()), k.why)),
    );
    assert!(
        bad.is_empty(),
        "a `new` line is a list and a shader disagreeing where `KNOWN` does not say \
         so; a `stale` one is a waiver excusing nothing:\n{}",
        bad.join("\n"),
    );
}

/// Every entry point the shaders declare is built into some pipeline.
///
/// The class-level form of "the table covers every pipeline": a record's `entries` is
/// the shaders' own answer about what it declares, so an entry point no [`Case`] names
/// is one the engine compiles and never runs — and nothing else would say so.
#[test]
fn every_entry_point_declared_is_built_into_some_pipeline() {
    let mut missing: Vec<String> = Vec::new();
    for r in spaces() {
        let table = table(r);
        let bound: Vec<EntryPoint> = table
            .cases
            .iter()
            .flat_map(|c| c.entries.iter().copied())
            .collect();
        for record in &table.records {
            for ep in record.entries {
                if !bound.contains(ep) {
                    missing.push(format!(
                        "resid={}: `{}`'s `{}`",
                        r.on(),
                        record.what,
                        ep.name,
                    ));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these entry points are declared and built into no pipeline here — either the \
         engine never runs them, or this table stopped following it:\n{}",
        missing.join("\n"),
    );
}
