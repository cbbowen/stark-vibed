//! Dev harness: replay a `.stark` file and diff the render with and without its
//! final stroke, to see exactly what that stroke changed. Usage:
//!
//! ```sh
//! cargo run -p stark-engine --example repro_bleed -- <file.stark> [out_dir] [--synth|--identity]
//! ```
//!
//! Prints the final stroke's brush and path stats, per-channel diff statistics
//! (including a replay-determinism control and the same stroke at decimated
//! control-point densities), and writes `with.png` / `without.png` / `diff.png`
//! (amplified) to `out_dir` (default: alongside the input).
//!
//! `--synth` replaces everything under the final stroke with a perfectly uniform
//! `Fill`, so any diff is numerical rather than smoothing of real coat texture;
//! `--identity` strips every axis off the final stroke while keeping it on the
//! stamp-loop path, so any diff is the loop's own rewrite drift.

mod common;

use stark_model::Srgb;
use std::io::BufWriter;
use std::path::PathBuf;

use stark_engine::RgbaImage;
use stark_engine::command::ViewCommand;
use stark_engine::headless_engine;
use stark_model::DocumentFile;
use stark_model::document::{ActionKind, FillOp, SelectionShape};
use stark_model::geom::{Extent2, Vec2};

fn save_png(path: &PathBuf, img: &RgbaImage) {
    let file = std::fs::File::create(path).expect("create png");
    let mut enc = png::Encoder::new(BufWriter::new(file), img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().expect("png header");
    w.write_image_data(&img.pixels).expect("png data");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .expect("usage: repro_bleed <file.stark> [out_dir]"),
    );
    let out_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| input.parent().expect("input has a parent").to_path_buf());

    let synth = std::env::args().any(|a| a == "--synth");
    let identity = std::env::args().any(|a| a == "--identity");
    let bytes = std::fs::read(&input).expect("read input");
    let mut doc = DocumentFile::from_bytes(&bytes).expect("parse .stark");
    if identity {
        // Strip every axis off the final stroke but keep it on the stamp-loop path
        // (a whisper of `charge` forces the loop): the loop then rewrites every
        // texel it covers with what should be the identity, and any diff against
        // `without` is the loop's own rewrite drift — nothing to do with bleed.
        let last = doc
            .actions
            .iter()
            .rposition(|a| matches!(a.kind, ActionKind::CommitStroke(_)))
            .expect("no stroke in file");
        let ActionKind::CommitStroke(rec) = &mut doc.actions[last].kind else {
            unreachable!()
        };
        rec.brush.wet_mut().expect("a wet brush").dynamics.bleed = 0.0;
        rec.brush.wet_mut().expect("a wet brush").dynamics.add = 0.0;
        rec.brush.wet_mut().expect("a wet brush").dynamics.charge = 1e-4;
        println!("--identity: final stroke reduced to the bare stamp loop");
    }
    if synth {
        // Replace everything under the final stroke with a PERFECTLY uniform
        // coat — a Fill lays constant tiles — so any diff the stroke leaves is
        // numerical, not smoothing of real coat texture.
        let last = doc
            .actions
            .iter()
            .rposition(|a| matches!(a.kind, ActionKind::CommitStroke(_)))
            .expect("no stroke in file");
        let stroke = doc.actions[last].clone();
        let ActionKind::CommitStroke(rec) = &stroke.kind else {
            unreachable!()
        };
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for cp in &rec.path {
            lo.x = lo.x.min(cp.pos.x);
            lo.y = lo.y.min(cp.pos.y);
            hi.x = hi.x.max(cp.pos.x);
            hi.y = hi.y.max(cp.pos.y);
        }
        let pad = Vec2::splat(rec.brush.size * 2.0 + 32.0);
        let mut fill = stroke.clone();
        fill.id.lamport = stroke.id.lamport.saturating_sub(1);
        fill.kind = ActionKind::Fill {
            layer: rec.layer,
            frame: stark_model::geom::IVec2::ZERO,
            op: FillOp::new(
                SelectionShape::rect_from_corners(lo - pad, hi + pad),
                0.0,
                Srgb::new([0.45, 0.40, 0.38]),
                1.0,
            ),
        };
        doc.actions = vec![fill, stroke];
        println!("--synth: uniform fill coat + the file's final stroke");
    }
    let doc = doc;
    let bytes = doc.to_bytes().expect("re-encode");
    println!("actions: {}", doc.actions.len());
    for (i, a) in doc.actions.iter().enumerate() {
        println!("  #{i}: {}", a.kind.label());
    }

    // The last stroke is the one under study.
    let last = doc
        .actions
        .iter()
        .rposition(|a| matches!(a.kind, ActionKind::CommitStroke(_)))
        .expect("no stroke in file");
    let ActionKind::CommitStroke(rec) = &doc.actions[last].kind else {
        unreachable!()
    };
    let b = &rec.brush;
    println!(
        "last stroke: action #{last}, {} control points, radius {}, drain {}/radius, \
         flow {}, add {}, lift {}, deposit {}, charge {}, bleed {}, opacity {}",
        rec.path.len(),
        b.size,
        b.drain,
        b.wet().expect("a wet brush").flow,
        b.wet().expect("a wet brush").dynamics.add,
        b.wet().expect("a wet brush").dynamics.lift,
        b.wet().expect("a wet brush").dynamics.deposit,
        b.wet().expect("a wet brush").dynamics.charge,
        b.wet().expect("a wet brush").dynamics.bleed,
        b.effect.opacity(),
    );

    // The stroke's bounding box (control points + radius), for the viewport.
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for cp in &rec.path {
        lo.x = lo.x.min(cp.pos.x);
        lo.y = lo.y.min(cp.pos.y);
        hi.x = hi.x.max(cp.pos.x);
        hi.y = hi.y.max(cp.pos.y);
    }
    let margin = b.size * 1.5 + 8.0;
    let center = (lo + hi) * 0.5;
    let size = Extent2 {
        width: (((hi.x - lo.x) + 2.0 * margin).ceil() as u32).clamp(64, 2048),
        height: (((hi.y - lo.y) + 2.0 * margin).ceil() as u32).clamp(64, 2048),
    };
    println!(
        "stroke bbox: ({:.1},{:.1})..({:.1},{:.1}), viewport {}x{} centered on ({:.1},{:.1})",
        lo.x, lo.y, hi.x, hi.y, size.width, size.height, center.x, center.y
    );

    let render = |bytes: &[u8]| -> RgbaImage {
        let mut engine = pollster::block_on(headless_engine(wgpu::TextureFormat::Rgba8Unorm, size))
            .expect("headless engine");
        let file = stark_model::DocumentFile::from_bytes(bytes).expect("decode");
        common::settle(&mut engine, &file);
        engine.load_document(&file).expect("load + replay");
        engine.process(ViewCommand::CenterOn(center));
        engine.render_to_image()
    };

    let (pmin, pmax) = rec.path.iter().fold((f32::MAX, f32::MIN), |(lo, hi), cp| {
        (lo.min(cp.pressure), hi.max(cp.pressure))
    });
    println!("pressure range: {pmin:.3}..{pmax:.3}");

    // Path stats: how finely the input was fitted.
    let mut path_len = 0.0f32;
    for w in rec.path.windows(2) {
        path_len += (w[1].pos - w[0].pos).length();
    }
    println!(
        "path length {:.1} px over {} spans -> mean span {:.2} px",
        path_len,
        rec.path.len() - 1,
        path_len / (rec.path.len() - 1) as f32
    );

    let with = render(&bytes);
    let mut doc_without = doc.clone();
    doc_without.actions.remove(last);
    let without_bytes = doc_without.to_bytes().expect("re-encode");
    let without = render(&without_bytes);
    // Control: two fresh replays of the SAME doc must render bit-identically, or
    // every diff below is measuring replay nondeterminism, not the stroke.
    let without2 = render(&without_bytes);

    let diff_stats = |tag: &str, a: &RgbaImage, b: &RgbaImage| {
        let mut n = [0u64; 4]; // |Δ| exceeding 2, 5, 10, 20
        let mut pos = [0i64; 3];
        let mut neg = [0i64; 3];
        let mut changed = 0u64;
        let mut worst = 0i32;
        for (pa, pb) in a
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.pixels.as_chunks::<4>().0)
        {
            let d: Vec<i32> = (0..3).map(|i| pa[i] as i32 - pb[i] as i32).collect();
            let m = d.iter().map(|v| v.abs()).max().unwrap();
            if m > 0 {
                changed += 1;
                for i in 0..3 {
                    if d[i] > 0 {
                        pos[i] += d[i] as i64;
                    } else {
                        neg[i] -= d[i] as i64;
                    }
                }
            }
            worst = worst.max(m);
            for (slot, tol) in n.iter_mut().zip([2, 5, 10, 20]) {
                *slot += (m > tol) as u64;
            }
        }
        let total = (a.width * a.height) as f64;
        println!(
            "{tag}: {changed} px changed ({:.2}%), worst {worst}; >2: {:.2}%, >5: {:.2}%, \
             >10: {:.2}%, >20: {:.2}%",
            changed as f64 / total * 100.0,
            n[0] as f64 / total * 100.0,
            n[1] as f64 / total * 100.0,
            n[2] as f64 / total * 100.0,
            n[3] as f64 / total * 100.0,
        );
        println!(
            "{tag}: sum +/- per channel: r +{}/-{}, g +{}/-{}, b +{}/-{}",
            pos[0], neg[0], pos[1], neg[1], pos[2], neg[2]
        );
    };
    diff_stats("control(without vs without2)", &without, &without2);
    diff_stats("full", &with, &without);

    // The same stroke with its path decimated: same geometry (so the same
    // exposure integral), far fewer spans. If the loop composes exactly, the
    // diff against `without` matches the full-resolution one.
    for keep in [4usize, 16] {
        let mut doc_dec = doc.clone();
        let ActionKind::CommitStroke(r) = &mut doc_dec.actions[last].kind else {
            unreachable!()
        };
        let n = r.path.len();
        let mut kept: Vec<_> = r.path.iter().copied().step_by(keep).collect();
        if (n - 1) % keep != 0 {
            kept.push(r.path[n - 1]);
        }
        r.path = kept;
        println!("decimated 1/{keep}: {} control points", r.path.len());
        let dec = render(&doc_dec.to_bytes().expect("re-encode"));
        diff_stats(&format!("dec{keep}"), &dec, &without);
    }

    // Amplified diff image: 128 + 8×Δ per channel.
    let mut diff = with.clone();
    for (pd, (pa, pb)) in diff.pixels.as_chunks_mut::<4>().0.iter_mut().zip(
        with.pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(without.pixels.as_chunks::<4>().0),
    ) {
        for i in 0..3 {
            pd[i] = (128 + 8 * (pa[i] as i32 - pb[i] as i32)).clamp(0, 255) as u8;
        }
        pd[3] = 255;
    }
    save_png(&out_dir.join("with.png"), &with);
    save_png(&out_dir.join("without.png"), &without);
    save_png(&out_dir.join("diff.png"), &diff);
    println!(
        "wrote with.png / without.png / diff.png to {}",
        out_dir.display()
    );
}
