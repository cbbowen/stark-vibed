//! Every entry point the shaders declare, against the pipelines the engine builds
//! (§6.10). Nothing here changes the engine.
//!
//! This table also held every hand-written slot list against the shader's own fold.
//! There are no such lists left — a layout is derived from the entry points that bind
//! it — so what remains is the question a derivation cannot answer: whether an entry
//! point is built into anything at all.

use std::collections::BTreeSet;

use stark_shaders::{EntryPoint, Lane, Resid};

/// One pipeline the engine creates, and the entry points it is built from.
///
/// A render pipeline is read as one set: its layout has to satisfy the vertex and
/// fragment stages together.
struct Case {
    /// The label `desc` gives `wgpu`, shortened.
    what: &'static str,
    entries: Vec<EntryPoint>,
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
fn space_table(r: Resid) -> Table {
    match r {
        // Exhaustive either way, with no gate of its own: `Resid::With` is generated
        // only where the build linked the pigment variant.
        #[cfg(feature = "mixbox")]
        Resid::With => {
            let (m, b, f) = (
                stark_shaders::media_mixbox(),
                stark_shaders::blend_mixbox(),
                stark_shaders::filter_mixbox(),
            );
            space_shaped(
                ("media_mixbox", m.entries, [m.vs_main, m.fs_main]),
                ("blend_mixbox", b.entries, [b.vs_main, b.fs_main]),
                (
                    "filter_mixbox",
                    f.entries,
                    [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
                ),
            )
        }
        Resid::Without => {
            let (m, b, f) = (
                stark_shaders::media_oklab(),
                stark_shaders::blend_oklab(),
                stark_shaders::filter_oklab(),
            );
            space_shaped(
                ("media_oklab", m.entries, [m.vs_main, m.fs_main]),
                ("blend_oklab", b.entries, [b.vs_main, b.fs_main]),
                (
                    "filter_oklab",
                    f.entries,
                    [f.vs_main, f.fs_main, f.fs_tile, f.fs_blur_decode],
                ),
            )
        }
    }
}

/// One space's record, as [`space_shaped`] takes it: its name, everything it declares,
/// and the entry points the pipelines below name.
type Shader<const N: usize> = (&'static str, &'static [EntryPoint], [EntryPoint; N]);

/// The five pipelines the two spaces have the same shape of — one media pass, one
/// blend, and the filter's three fragment entry points over one layout.
fn space_shaped(media: Shader<2>, blend: Shader<2>, filter: Shader<4>) -> Table {
    let [vs, fs_main, fs_tile, fs_blur_decode] = filter.2;
    let one = |what, fs| Case {
        what,
        entries: vec![vs, fs],
    };
    Table {
        cases: vec![
            Case {
                what: "media",
                entries: media.2.to_vec(),
            },
            Case {
                what: "blend",
                entries: blend.2.to_vec(),
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

    let mut all = vec![
        Case {
            what: "composite",
            entries: vec![composite.vs_main, composite.fs_main],
        },
        Case {
            what: "matte",
            entries: vec![matte.vs_main, matte.fs_main],
        },
        Case {
            what: "overlay",
            entries: vec![ov.vs_main, ov.fs_main],
        },
        Case {
            what: "guides",
            entries: vec![gu.vs_main, gu.fs_main],
        },
        Case {
            what: "resolve",
            entries: vec![re.vs_main, re.fs_main],
        },
        Case {
            what: "blur fft",
            entries: vec![bl.fft_both],
        },
        Case {
            what: "blur kernel fft",
            entries: vec![bl.fft_one],
        },
        Case {
            what: "blur make kernel",
            entries: vec![bl.make_kernel],
        },
        Case {
            what: "blur apply kernel",
            entries: vec![bl.apply_kernel],
        },
        Case {
            what: "fill",
            entries: vec![fi.vs_main, fi.fs_main],
        },
        Case {
            what: "merge",
            entries: vec![me.vs_main, me.fs_main],
        },
        Case {
            what: "slab expand",
            entries: vec![sl.vs_main, sl.fs_expand],
        },
        Case {
            what: "slab store",
            entries: vec![sl.vs_main, sl.fs_store],
        },
        Case {
            what: "transform parcel",
            entries: vec![tr.vs_quad, tr.fs_parcel],
        },
        Case {
            what: "transform mask",
            entries: vec![tr.vs_quad, tr.fs_mask],
        },
        Case {
            what: "transform parcel gated",
            entries: vec![tr.vs_gated, tr.fs_parcel_gated],
        },
        Case {
            what: "transform mask gated",
            entries: vec![tr.vs_gated, tr.fs_mask_gated],
        },
        Case {
            what: "transform combine",
            entries: vec![tr.vs_fill, tr.fs_combine],
        },
        Case {
            what: "transform mask base",
            entries: vec![tr.vs_fill, tr.fs_mask_base],
        },
        Case {
            what: "selection",
            entries: vec![se.vs_main, se.fs_main],
        },
        Case {
            what: "selection region",
            entries: vec![mr.vs_main, mr.fs_main],
        },
        Case {
            what: "sweep",
            entries: vec![plain.vs_main, plain.fs_main],
        },
        Case {
            what: "sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_main],
        },
        Case {
            what: "sweep levels",
            entries: vec![plain.vs_main, plain.fs_levels],
        },
        Case {
            what: "erase sweep",
            entries: vec![plain.vs_main, plain.fs_erase],
        },
        Case {
            what: "erase sweep ceiling",
            entries: vec![ceiling.vs_main, ceiling.fs_erase],
        },
        Case {
            what: "integrate",
            entries: vec![ig.vs_main, ig.fs_main],
        },
        Case {
            what: "erase",
            entries: vec![er.vs_main, er.fs_main],
        },
        Case {
            what: "dynamics composite",
            entries: vec![composite.vs_main, composite.fs_raw],
        },
        Case {
            what: "dynamics snapshot",
            entries: vec![dy.snapshot],
        },
        Case {
            what: "dynamics bleed weight",
            entries: vec![dy.bleed_weight],
        },
        Case {
            what: "dynamics exchange",
            entries: vec![dy.exchange],
        },
        Case {
            what: "dynamics bake",
            entries: vec![dy.bake],
        },
        Case {
            what: "dynamics deposit",
            entries: vec![dy.deposit],
        },
        Case {
            what: "dynamics cell hoist",
            entries: vec![dy.cell_hoist],
        },
        Case {
            what: "dynamics deposit coarse",
            entries: vec![dy.deposit_coarse],
        },
        Case {
            what: "dynamics settle",
            entries: vec![dy.settle],
        },
        Case {
            what: "liquify snapshot field",
            entries: vec![li.snapshot_field],
        },
        Case {
            what: "liquify warp",
            entries: vec![li.warp],
        },
        Case {
            what: "liquify warp apply",
            entries: vec![li.warp_apply],
        },
        Case {
            what: "dynamics slice",
            entries: vec![sc.vs_main, sc.fs_main],
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
        // Both builds of the sweep. Not "the plain build alone, since the variants
        // declare the same names": a name is not what is checked here, a *value* is,
        // and the two builds' `fs_main` differ in the targets they write.
        Record {
            what: "stamp",
            entries: plain.entries,
        },
        Record {
            what: "stamp ceiling",
            entries: ceiling.entries,
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
    let space = space_table(r);
    all.extend(space.cases);
    records.extend(space.records);
    Table {
        cases: all,
        records,
    }
}

/// The colour spaces this build has, which the check below runs over.
fn spaces() -> Vec<Resid> {
    #[cfg(feature = "mixbox")]
    let spaces = vec![Resid::Without, Resid::With];
    #[cfg(not(feature = "mixbox"))]
    let spaces = vec![Resid::Without];
    spaces
}

/// One entry point an artifact declares that no pipeline here names, and why.
///
/// Keyed on the colour space too: an artifact only one space links carries its dead
/// entry point only there.
struct Unbuilt {
    resid: bool,
    /// The record, as [`table`] names it.
    what: &'static str,
    entry: &'static str,
    why: &'static str,
}

impl Unbuilt {
    fn at(&self) -> (bool, &'static str, &'static str) {
        (self.resid, self.what, self.entry)
    }
}

/// The entry points compiled into an artifact and built into nothing.
///
/// The sweep is linked along the ceiling axis as well as the residual, and `fs_levels`
/// is not `@if(ceiling)`-gated — it writes the lane and nothing else, so the same
/// function is right in both builds and the gate would say nothing. The stamp loop's
/// per-segment draw of the lane runs the **plain** module's (`swept::build_swept_kit`),
/// so the ceiling build translates a second copy that no pipeline names.
///
/// Costs a translation per ceiling variant and nothing else; gating it in the WESL
/// would retire these rows, which is what makes them worth stating.
const UNBUILT: &[Unbuilt] = &[
    Unbuilt {
        resid: false,
        what: "stamp ceiling",
        entry: "fs_levels",
        why: "not `@if(ceiling)`-gated, and the levels pipeline runs the plain build's",
    },
    Unbuilt {
        resid: true,
        what: "stamp ceiling",
        entry: "fs_levels",
        why: "the same, in the pigment space",
    },
];

/// Every entry point the shaders declare is built into some pipeline, **except** the
/// declared few.
///
/// The class-level form of "the table covers every pipeline": a record's `entries` is
/// the shaders' own answer about what it declares, so an entry point no [`Case`] names
/// is one the engine compiles and never runs — and nothing else would say so.
///
/// An **exact** set against [`UNBUILT`], so a row that has stopped excusing anything
/// fails as loudly as an entry point nobody triaged — which is how an allow-list turns
/// into the place the next mistake hides.
///
/// Every *build* of a shader is a record here, not one per shader. The comparison is
/// over `EntryPoint` values and a variant's differ — the sweep's `fs_main` writes
/// `[0, 1]` plain and `[0, 1, 3]` with the ceiling lane — so "the variants declare the
/// same names" would have been an answer to a question nobody asked.
#[test]
fn every_entry_point_declared_is_built_into_some_pipeline() {
    let mut found: BTreeSet<(bool, &'static str, &'static str)> = BTreeSet::new();
    for r in spaces() {
        let table = table(r);
        for case in &table.cases {
            assert!(
                !case.entries.is_empty(),
                "`{}` names no entry point, so it covers nothing",
                case.what,
            );
        }
        let bound: Vec<EntryPoint> = table
            .cases
            .iter()
            .flat_map(|c| c.entries.iter().copied())
            .collect();
        for record in &table.records {
            for ep in record.entries {
                if !bound.contains(ep) {
                    found.insert((r.on(), record.what, ep.name));
                }
            }
        }
    }
    let here = |resid: bool| spaces().iter().any(|r| r.on() == resid);
    let declared: BTreeSet<_> = UNBUILT
        .iter()
        .map(Unbuilt::at)
        .filter(|(resid, ..)| here(*resid))
        .collect();
    let say =
        |(resid, what, entry): &(bool, &str, &str)| format!("resid={resid}: `{what}`'s `{entry}`");
    let mut bad: Vec<String> = found
        .difference(&declared)
        .map(|w| format!("new:   {}", say(w)))
        .collect();
    bad.extend(
        UNBUILT
            .iter()
            .filter(|u| here(u.resid) && !found.contains(&u.at()))
            .map(|u| format!("stale: {} — declared as {}", say(&u.at()), u.why)),
    );
    assert!(
        bad.is_empty(),
        "a `new` line is an entry point this build compiles and no pipeline names — \
         either the engine never runs it, or this table stopped following it; a \
         `stale` one is a row excusing nothing:\n{}",
        bad.join("\n"),
    );
}
