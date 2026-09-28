//! The blessed pictures (§9), and the two switches that decide what a comparison
//! against one means.

// The siblings of this module, and the colours and corpus beside them: every
// helper here is re-exported from `common`, so one glob reaches all of them.
use super::*;
use stark_engine::RgbaImage;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Set to `1` to run the golden tests but not *compare* their output — see
/// [`assert_golden`]. For adapters other than the one the goldens were blessed on.
const SKIP_GOLDEN: &str = "STARK_SKIP_GOLDEN";

/// Set to `1` to compare every golden at tolerance 0, whatever the test asked for.
/// The gate for a change claimed not to move a pixel: a per-test tolerance absorbs
/// exactly the drift such a change must not have.
const GOLDEN_EXACT: &str = "STARK_GOLDEN_EXACT";

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// Compare `img` against the committed golden `tests/golden/<name>.png`.
///
/// If the golden file is absent it is created and the check passes — so
/// **deleting a golden re-blesses it** on the next run. Goldens are
/// GPU-dependent; cross-adapter runs may need re-blessing (§9).
///
/// Which is why [`SKIP_GOLDEN`] exists: a committed golden can only match the one
/// adapter it was blessed on, so a CI runner on a different adapter would fail on
/// pixels rather than on behaviour. Only the *pixel comparison* is dropped — the
/// stroke is still rendered, so shader compilation, validation errors and panics are
/// caught, and what does not depend on the adapter is still asserted: the render is
/// not one flat colour, and it is the size the committed golden is. An absent golden
/// is **not** blessed under the flag, since the adapter it would come from is the one
/// the flag distrusts; there, only the first of the two holds.
///
/// The weakest form of the corpus battery's `check_leaves_a_mark` — weakest because there is
/// no before-render to difference against, so a case that paints over an undercoat is
/// covered by the corpus battery and not by this — and here for that check's own
/// reason: the tests whose whole assertion is this call reported `ok` under the flag
/// for having not panicked.
pub fn assert_golden(name: &str, img: &RgbaImage, tol: u8) {
    // Above the [`SKIP_GOLDEN`] branch, and so one of the two things that round still
    // says about the picture rather than about the process surviving it.
    assert!(
        !is_uniform(img),
        "golden {name}: the {}x{} render is one flat colour {:?}, so there is no \
         picture in it to compare",
        img.width,
        img.height,
        img.pixels.first_chunk::<4>().copied().unwrap_or_default()
    );

    let dir = golden_dir();
    fs::create_dir_all(&dir).expect("create golden dir");
    let path = dir.join(format!("{name}.png"));
    let skip = env_flag(SKIP_GOLDEN);

    if !path.exists() {
        // Blessing writes *this* adapter's pixels as the committed truth, which is the
        // one thing the flag says they may not be taken for.
        if skip {
            eprintln!("golden {name}: absent, and {SKIP_GOLDEN}=1 does not bless");
            return;
        }
        write_png(&path, img);
        eprintln!("blessed new golden: {}", path.display());
        return;
    }

    let golden = read_png(&path);
    assert_eq!(
        (img.width, img.height),
        (golden.width, golden.height),
        "golden {name}: size mismatch"
    );
    if skip {
        eprintln!(
            "golden {name}: {}x{} rendered, not one flat colour; pixel comparison \
             skipped ({SKIP_GOLDEN}=1)",
            img.width, img.height
        );
        return;
    }

    // **Worst-texel**, which is the same statistic the corpus battery holds every
    // other one of its checks to (`corpus::Report::check`) and is here for that
    // comparator's stated reason: "steps and seams are loud in the maximum and quiet
    // in the average".
    //
    // Counting a *fraction* instead passes while up to 1% of the frame differs by any
    // amount whatever. 1% of a 256² golden is 655 texels — a disc of radius 14, the
    // size of the lift-end ring, the settle crease and the stranded glob, each of
    // which has a bespoke test in `tests/engine/dynamics.rs` because a golden did not catch
    // it. A fraction also hides ordinary drift: `corpus_curve` and `corpus_bleed` sat
    // 456 and 559 texels out (max 10 and 8) against un-re-blessed goldens, silently,
    // for as long as it stayed under 1%.
    //
    // What the fraction was really absorbing is visible in that measurement's own
    // shape — 2100 texels at 1 level, 3400 at 2, decaying to 2 at 10, scattered at
    // ~3% density over the stroke. Legitimate numeric drift is a *decaying tail over
    // a region*; an artifact is a *bump at high magnitude*. Only the maximum tells
    // them apart, so only the maximum is read, and `tol` is the whole bound.
    let (d, i) = texel_deltas(img, &golden)
        .enumerate()
        .fold(
            (0u8, 0usize),
            |worst, (i, d)| if d > worst.0 { (d, i) } else { worst },
        );
    let tol = if env_flag(GOLDEN_EXACT) { 0 } else { tol };
    if d > tol {
        let actual = path.with_extension("actual.png");
        write_png(&actual, img);
        panic!(
            "golden {name} mismatch: worst {d} > tol {tol}, at ({}, {}); wrote {}",
            i as u32 % img.width,
            i as u32 / img.width,
            actual.display()
        );
    }
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Through [`RgbaImage::to_png`] rather than an encoder of its own: that is the shipped
/// export path (§15.6), so a golden is blessed in the bytes an export would write.
pub fn write_png(path: &Path, img: &RgbaImage) {
    fs::write(path, img.to_png().expect("encode png")).expect("write png");
}

fn read_png(path: &Path) -> RgbaImage {
    let decoder = png::Decoder::new(BufReader::new(fs::File::open(path).expect("open png")));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(
        info.color_type,
        png::ColorType::Rgba,
        "golden must be RGBA8"
    );
    buf.truncate(info.buffer_size());
    RgbaImage::new(info.width, info.height, buf)
}
