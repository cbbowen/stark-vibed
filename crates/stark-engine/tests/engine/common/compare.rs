//! Reading a texel, and holding two renders against each other.

// The siblings of this module, and the colours and corpus beside them: every
// helper here is re-exported from `common`, so one glob reaches all of them.
use super::*;
use stark_engine::RgbaImage;
use stark_model::geom::Vec2;
use std::fmt;

/// One channel of a rendered texel back to the 8-bit sRGB code it *means* — the
/// decode is the target's own transfer, the encode is sRGB's, and the clamp is the
/// 8-bit store's.
///
/// Per channel, so a transfer that changes primaries is out of its reach:
/// `tests/engine/wide.rs` is where a change of primaries is measured.
pub fn srgb_code(transfer: stark_engine::Transfer, v: f32) -> i32 {
    use stark_engine::Transfer;
    use stark_model::color::{linear_to_srgb, srgb_to_linear};
    let lin = match transfer {
        Transfer::Linear => v,
        Transfer::ExtendedSrgb => srgb_to_linear(v),
        Transfer::Srgb => srgb_to_linear(v.clamp(0.0, 1.0)),
        p3 => unreachable!("{p3:?} changes primaries, which one channel cannot undo"),
    };
    (linear_to_srgb(lin.clamp(0.0, 1.0)) * 255.0).round() as i32
}

/// The centre pixel — where the suites' standard stroke crosses.
pub fn center(img: &RgbaImage) -> [u8; 4] {
    img.pixel(img.width / 2, img.height / 2)
}

/// Where canvas point `canvas` lands on a [`SIZE`] screen, at the default 1:1 view
/// centred on the origin.
pub fn screen_of(canvas: Vec2) -> (u32, u32) {
    let half = Vec2::new(SIZE.width as f32, SIZE.height as f32) * 0.5;
    let p = canvas + half;
    (p.x as u32, p.y as u32)
}

/// The RGB at canvas point `canvas`, as signed channels a difference can be taken in.
///
/// The origin comes off **this image's** centre rather than [`SIZE`]'s, so an export
/// or a sized engine's render can be probed by the same canvas coordinate as a
/// default one.
pub fn texel(img: &RgbaImage, canvas: Vec2) -> [i32; 3] {
    let p = canvas + Vec2::new(img.width as f32, img.height as f32) * 0.5;
    rgb(img.pixel(p.x as u32, p.y as u32))
}

/// Two colors agree to within `tol` in every channel.
pub fn near(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

/// Whether a pixel reads as mat-board black — what a matte paints in (§15.4).
pub fn is_dark(c: [u8; 4]) -> bool {
    c[0] < 60 && c[1] < 60 && c[2] < 60
}

/// A pixel's three color channels, signed — what [`leads`] and [`apart`] both compare.
pub fn rgb(c: [u8; 4]) -> [i32; 3] {
    [c[0] as i32, c[1] as i32, c[2] as i32]
}

/// The worst per-channel distance between two texels.
pub fn apart(a: [i32; 3], b: [i32; 3]) -> i32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).max().unwrap()
}

/// Which channel a probe is asking about.
///
/// An enum rather than an index because half the files that probe also declare a
/// brush color named `RED`, and the two would collide on import.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lead {
    Red,
    Green,
    Blue,
}

/// Whether channel `lead` of `c` exceeds **both** the others by at least `margin`.
///
/// **The one predicate under every "is there paint here?" in the suite.** It was
/// written out twelve times across nine test files, under four names — `is_red`,
/// `is_painted`, `red_dominant`, and the same three for the other channels — varying
/// along two axes the copies did not separate: which channel leads, and by how much.
/// The margins that resulted — 30, 40 and 60 — were not a considered range but the
/// order the files were written in, and no copy said which number was a threshold and
/// which was a leftover.
///
/// The margin stays a parameter, because it is not a formatting detail: about half
/// the call sites read `!`, so raising it weakens an assertion in one file and
/// strengthens it in the next. [`MARGIN`], [`MARGIN_FLAT`] and [`MARGIN_LIT`] are the
/// three the suite distinguishes; a file wanting a fourth should say why in a `const`
/// of its own rather than in a literal.
///
/// A *margin* rather than a level, because the canvas is not black: what separates
/// paint from paper is the hue, and a level would need re-tuning for every substrate
/// the suite paints on.
pub fn leads(c: [i32; 3], lead: Lead, margin: i32) -> bool {
    let i = lead as usize;
    (0..3).all(|k| k == i || c[i] > c[k] + margin)
}

/// The separation the compositor guarantees between the suite's paint and its paper.
pub const MARGIN: i32 = 30;

/// The separation asked for by the files that probe **flat** paint at a point.
///
/// Above [`MARGIN`] not because those renders are noisier but because their negative
/// assertions are the load-bearing ones: a fill test's job is largely to show paint
/// did *not* reach somewhere, and a gradient's far stop is a blend of two of these
/// colors that has to fail the test for the other one.
pub const MARGIN_FLAT: i32 = 40;

/// The separation asked for by the files that probe **lit** paint at a point.
///
/// Wide on purpose, and not because the substrate under it is warm: under the reference
/// light (§6.3) [`PAPER`] is achromatic and carries no dominance at all, while the
/// suite's red paint dominates by ~210 and [`BG`] by ~180. What 60 buys is that these
/// files state the separation that matters rather than the one that just passes — a
/// warm light, which is one view setting away, would push a neutral substrate to ~33
/// levels of false dominance on its own.
pub const MARGIN_LIT: i32 = 60;

/// Whether a pixel reads as the suite's red paint rather than as canvas, at [`MARGIN`].
///
/// One copy with one margin, shared by every suite that asks "is there paint here?" —
/// so a render change that shifts channel separation moves every test the same
/// distance instead of being patched file by file.
pub fn red_dominant(c: [u8; 4]) -> bool {
    leads(rgb(c), Lead::Red, MARGIN)
}

/// [`red_dominant`] for the other color the suite paints in.
pub fn green_dominant(c: [u8; 4]) -> bool {
    leads(rgb(c), Lead::Green, MARGIN)
}

/// [`red_dominant`] asked of a **canvas point** on a [`SIZE`] render, at [`MARGIN_FLAT`].
pub fn painted(img: &RgbaImage, canvas: Vec2) -> bool {
    leads(texel(img, canvas), Lead::Red, MARGIN_FLAT)
}

/// The worst per-channel difference between two texels.
pub fn texel_delta(a: [u8; 4], b: [u8; 4]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(&x, y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

/// [`texel_delta`] at every texel of two renders of the same size.
///
/// **The suite's one diff kernel.** The same four lines were written out ten times —
/// in this module three times, and again in the composite, tooth and modulation
/// suites — every copy agreeing on the statistic and differing only in what it did
/// with it afterwards: a fraction, a worst, a hit set, a per-texel ink measure. They
/// are all this iterator plus a fold.
///
/// Panics on a size mismatch, which is not a question any caller wants answered
/// texel by texel.
pub fn texel_deltas<'a>(a: &'a RgbaImage, b: &'a RgbaImage) -> impl Iterator<Item = u8> + 'a {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "image size mismatch"
    );
    a.pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.pixels.as_chunks::<4>().0)
        .map(|(&pa, &pb)| texel_delta(pa, pb))
}

/// The fraction of pixels that differ **at all**, and the worst per-channel
/// difference anywhere — in that order.
///
/// The two answer different questions and most callers want both: the fraction
/// separates a contiguous seam from a speck, the worst says how far the seam went.
/// [`frac_exceeding`] is the same fraction taken above a threshold.
pub fn diff_fraction(a: &RgbaImage, b: &RgbaImage) -> (f64, u8) {
    let mut bad = 0u64;
    let mut worst = 0u8;
    for d in texel_deltas(a, b) {
        worst = worst.max(d);
        bad += u64::from(d > 0);
    }
    (bad as f64 / (a.width * a.height) as f64, worst)
}

/// Fraction of pixels whose maximum per-channel difference *exceeds* `tol`. Unlike the
/// per-pixel worst, this distinguishes a contiguous seam (a band of many significantly-
/// different pixels) from a handful of isolated precision specks.
pub fn frac_exceeding(a: &RgbaImage, b: &RgbaImage, tol: u8) -> f64 {
    let bad = texel_deltas(a, b).filter(|&d| d > tol).count();
    bad as f64 / (a.width * a.height) as f64
}

/// True if `a` and `b` match within `tol` per channel for every pixel.
///
/// For a *question* about two renders. A test making the claim wants
/// [`assert_images`], whose failure says how far apart they are and where.
pub fn images_match(a: &RgbaImage, b: &RgbaImage, tol: u8) -> bool {
    texel_deltas(a, b).all(|d| d <= tol)
}

/// The first few texels more than `tol` apart, as `(x,y) before -> after [delta]`.
fn differing_sites(a: &RgbaImage, b: &RgbaImage, tol: u8) -> Vec<String> {
    texel_deltas(a, b)
        .enumerate()
        .filter(|&(_, d)| d > tol)
        .take(8)
        .map(|(i, d)| {
            let j = i * 4;
            format!(
                "({},{}) {:?} -> {:?} [{d}]",
                i as u32 % a.width,
                i as u32 / a.width,
                &a.pixels[j..j + 4],
                &b.pixels[j..j + 4]
            )
        })
        .collect()
}

/// Assert every texel of `a` and `b` is within `tol` per channel. `tol` of 0 is
/// byte-exact.
///
/// **The suite's one image assertion.** `assert!(images_match(..))` reports that two
/// renders differ and nothing else, so every investigation of one began by finding
/// out whether the drift was a seam, a speck or the whole frame. This says so: the
/// worst channel delta, the share of the frame past `tol`, and the first few sites.
///
/// `what` takes a literal or a `format_args!`, so a caller naming a case in the
/// message does not have to allocate one.
#[track_caller]
pub fn assert_images(a: &RgbaImage, b: &RgbaImage, tol: u8, what: impl fmt::Display) {
    let mut worst = 0u8;
    let mut over = 0u64;
    for d in texel_deltas(a, b) {
        worst = worst.max(d);
        over += u64::from(d > tol);
    }
    if worst <= tol {
        return;
    }
    panic!(
        "{what}: images differ by {worst} > tol {tol}, over {:.4}% of the frame\n\
         first sites: {}",
        over as f64 / (a.width * a.height) as f64 * 100.0,
        differing_sites(a, b, tol).join("; ")
    );
}

/// Assert `a` and `b` are more than `tol` apart **somewhere** — [`assert_images`]'s
/// negative, for a test whose claim is that something reached the canvas.
#[track_caller]
pub fn assert_images_differ(a: &RgbaImage, b: &RgbaImage, tol: u8, what: impl fmt::Display) {
    let worst = texel_deltas(a, b).max().unwrap_or(0);
    assert!(
        worst > tol,
        "{what}: the two renders agree to within {tol} (worst channel delta {worst}), \
         so nothing reached the canvas"
    );
}

/// True if every texel of `img` is the same RGBA value — a render with no picture in
/// it, whatever colour the flat is. An empty image counts as uniform.
///
/// The single-image half of what [`diff_fraction`] answers for a pair, and the one
/// claim about a picture that holds on *any* adapter: what a stroke's texels come out
/// to is the driver's business, that they are not all one value is not.
pub fn is_uniform(img: &RgbaImage) -> bool {
    let texels = img.pixels.as_chunks::<4>().0;
    let Some((first, rest)) = texels.split_first() else {
        return true;
    };
    rest.iter().all(|t| t == first)
}
