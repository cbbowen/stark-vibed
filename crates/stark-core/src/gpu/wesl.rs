//! Checks across the CPU↔shader boundary (§6.7, §7).
//!
//! Every `#[repr(C)]` uniform in this subsystem is one half of a pair the compiler
//! cannot see across: the shader decides how the lanes are *read*, and nothing on
//! this side knows what it decided. Whatever can be checked here should be, because
//! the failure is quiet — a wgpu validation error at best, a silently misread lane
//! at worst.

/// Pin a uniform struct's size against the WESL declaration it mirrors.
///
/// The convention had been to write the size into the doc comment — which is worse
/// than writing nothing, because a stale number reads as a verified one. Three had
/// drifted by the time this existed: `ViewUniform` said 32 and was 48,
/// `MediaUniform` said 80 and was 96 (`surf_m`, §18.1.2), and `GuideUniform` said
/// 240 and was 304 (the fisheye's second set of poles, §20.8). None of the three
/// was wrong in a way a pixel could show; all three were wrong in the one place a
/// maintainer goes to check.
///
/// So the number moves out of the prose and into the build. It does **not** prove
/// the lanes line up — only the shader-side declaration can say that, which is what
/// `the_stamp_struct_has_the_same_nine_lanes_on_both_sides` reads out of the WESL
/// source for `Stamp`, the one struct here whose mismatch would be silent rather
/// than a validation error. What this catches is the realistic change: a lane
/// appended to one side and not the other.
macro_rules! mirrors_wesl {
    ($t:ty, $bytes:expr) => {
        const _: () = assert!(
            std::mem::size_of::<$t>() == $bytes,
            concat!(
                stringify!($t),
                " is no longer the size of the WESL struct it mirrors",
            ),
        );
    };
}
pub(crate) use mirrors_wesl;
