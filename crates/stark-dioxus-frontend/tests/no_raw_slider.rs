//! Only `widgets.rs` draws a `.slider` track.
//!
//! The track is where a slider's fill, its step and the three events that end a drag
//! are written (`widgets::SliderTrack`). A range input wearing `.slider` anywhere else
//! is a second copy of those rules, and the copy that forgets one — an empty fill, a
//! preview that never lays down — still looks like a slider.

mod common;

use common::{read, rust_files};
use std::path::Path;

#[test]
fn only_widgets_draws_a_slider_track() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let widgets = Path::new("widgets.rs");

    // The check matches something, or it is a green test that matched nothing.
    assert!(
        !slider_lines(&read(&src.join(widgets))).is_empty(),
        "widgets.rs wears no `.slider`, so this check cannot see a track"
    );

    let mut found = Vec::new();
    for path in rust_files(&src) {
        let relative = path.strip_prefix(&src).expect("the walk stays under src");
        if relative == widgets {
            continue;
        }
        for line in slider_lines(&read(&path)) {
            found.push(format!("{}:{line}", path.display()));
        }
    }
    assert!(
        found.is_empty(),
        "a `.slider` track is drawn outside widgets.rs; use `Slider`, `PreviewSlider` or \
         `SliderTrack`:\n{}",
        found.join("\n")
    );
}

/// The line numbers of `source` whose `class:` literal wears `slider` among its classes.
/// Comment lines are prose and may say anything.
fn slider_lines(source: &str) -> Vec<usize> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .filter(|(_, line)| wears_slider(line))
        .map(|(n, _)| n + 1)
        .collect()
}

/// Whether some `class: "…"` on `line` names the `slider` class — alone or beside others,
/// so `"slider setting-slider"` counts and `"slider-row"` does not.
fn wears_slider(line: &str) -> bool {
    line.match_indices("class:").any(|(at, key)| {
        let rest = line[at + key.len()..].trim_start();
        rest.strip_prefix('"')
            .and_then(|literal| literal.split('"').next())
            .is_some_and(|classes| classes.split_whitespace().any(|class| class == "slider"))
    })
}
#[test]
fn a_track_is_recognized_alone_or_among_classes() {
    assert!(wears_slider(r#"            class: "slider","#));
    assert!(wears_slider(r#"    class: "slider setting-slider","#));
    assert!(wears_slider(r#"    class: "wide slider","#));
    assert!(!wears_slider(
        r#"        div { class: "slider-row marked","#
    ));
    assert!(!wears_slider(r#"    class: "{class}","#));
}
