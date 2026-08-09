//! The Credits dialog: other people's work that Stark ships, and the terms it comes
//! under.
//!
//! Off the ☰ menu rather than a panel or the command rail, on the rail's own rule
//! (main.rs): that menu is where infrequent commands live, and this is the least
//! frequent of them — read once, if ever.
//!
//! Two things it is deliberately not. It is not a settings dialog, so nothing here
//! is a control; and it is not a summary of a licence — the text is shown in full,
//! because a paraphrase of a permission notice is not the notice. MIT asks that the
//! notice travel with the copies, and a browser app has no `LICENSE` file for anyone
//! to find, so the dialog *is* where that obligation is met.

use dioxus::prelude::*;

use crate::icons::{self, icon};

/// Phosphor's licence, read from the copy vendored beside the icons it covers.
///
/// `include_str!` and not a string literal in this file, so the text on screen and
/// the text in the tree are the same bytes and cannot drift: the file next to the
/// SVGs is the one the licence asks us to ship, and this is a view of it rather than
/// a transcription of it.
const PHOSPHOR_LICENSE: &str = include_str!("../assets/icons/LICENSE-phosphor");

const IROH_SOCKET_LICENSE: &str = include_str!("../assets/licenses/LICENSE-iroh-socket");

const IROH_LICENSE: &str = include_str!("../assets/licenses/LICENSE-iroh");

/// The credits dialog, opened from the ☰ menu and dismissed by Done or by clicking
/// the backdrop (as the other dialogs are).
#[component]
pub fn CreditsModal(on_close: EventHandler<()>) -> Element {
    rsx! {
        div {
            class: "modal-backdrop",
            onclick: move |_| on_close.call(()),
            div {
                // Wider than the other dialogs: a licence has its own line breaks and
                // this is the width that respects them. Narrower and every paragraph
                // reflows into a ragged column — the text would still be exact, but it
                // would read as something we retyped rather than as the notice it is.
                class: "modal-dialog modal-wide",
                onclick: move |e| e.stop_propagation(),

                div { class: "modal-title", "Credits" }
                div { class: "modal-subtitle",
                    "Other people's work that Stark is built on, and the terms it comes under."
                }

                div {
                    class: "credits",

                    div { class: "modal-section-label", "SOFTWARE" }
                    Credit {
                        name: "Björn Ottosson",
                        url: "https://bottosson.github.io/posts/oklab",
                        description: "Oklab color space.",
                        license: "Public domain",
                    }
                    // Only where the code it attributes is actually in the build.
                    // This is the credit that has to track the `mixbox` feature most
                    // exactly of the three places that mention it: a build compiled
                    // without it contains no Mixbox code, and claiming a
                    // non-commercial licence it is not bound by would be its own kind
                    // of wrong.
                    {cfg!(feature = "mixbox").then(|| rsx! {
                        Credit {
                            name: "Secret Weapons",
                            url: "https://scrtwpns.com/mixbox",
                            description: "Mixbox color mixing.",
                            license: "Creative Commons Attribution-NonCommercial 4.0 International Public License",
                        }
                    })}
                    Credit {
                        name: "N0",
                        url: "https://www.iroh.computer",
                        description: "Iroh peer-to-peer networking stack.",
                        license: IROH_LICENSE,
                    }
                    Credit {
                        name: "Tailscale",
                        url: "https://tailscale.com",
                        description: "Parts of Iroh's socket implementation.",
                        license: IROH_SOCKET_LICENSE,
                    }

                    hr {}

                    div { class: "modal-section-label", "ICONS" }
                    Credit {
                        name: "Phosphor Icons",
                        url: "https://phosphoricons.com",
                        description: "Every glyph in Stark's panels, bars and menus.",
                        license: PHOSPHOR_LICENSE,
                    }

                    hr {}

                    div { class: "modal-section-label", "ENVIRONMENTS" }
                    Credit {
                        name: "Dimitrios Savva and Greg Zaal",
                        url: "https://polyhaven.com/a/ferndale_studio_11",
                        description: "Ferndale lighting environments.",
                        license: "CC0 1.0 Universal",
                    }
                    Credit {
                        name: "Greg Zaal and Jarod Guest",
                        url: "https://polyhaven.com/a/qwantani_dusk_2_puresky",
                        description: "Qwantani lighting environments.",
                        license: "CC0 1.0 Universal",
                    }
                    Credit {
                        name: "Greg Zaal and Jenelle van Heerden",
                        url: "https://polyhaven.com/a/bloem_hill_01",
                        description: "Bloem Hill lighting environments.",
                        license: "CC0 1.0 Universal",
                    }
                    Credit {
                        name: "Greg Zaal",
                        url: "https://polyhaven.com/a/kloofendal_overcast_puresky",
                        description: "Kloofendal lighting environments.",
                        license: "CC0 1.0 Universal",
                    }
                }

                div { class: "modal-actions",
                    button {
                        class: "btn btn-primary",
                        onclick: move |_| on_close.call(()),
                        {icon(icons::DONE)}
                        "Done"
                    }
                }
            }
        }
    }
}

/// One credited work: who made it, where it lives, what Stark uses it for, and its
/// licence in full.
///
/// A component rather than four lines of markup because the next entry should cost
/// one call — the dialog is a list that only ever grows, and an entry that has to be
/// laid out by hand is an entry that ends up laid out differently.
#[component]
fn Credit(
    name: String,
    url: String,
    description: String,
    /// The licence text verbatim. `&'static str`, because it is always an
    /// `include_str!` of a file in the tree — see [`PHOSPHOR_LICENSE`].
    license: &'static str,
) -> Element {
    rsx! {
        div { class: "credit",
            div { class: "credit-head",
                span { class: "credit-name", "{name}" }
                // Opens away from the app, which for a canvas holding unsaved work is
                // the only acceptable way out to a website: `_blank` keeps the drawing
                // where it is, and `noopener` keeps the new tab from reaching back into
                // the page that opened it.
                a {
                    class: "credit-link",
                    href: "{url}",
                    target: "_blank",
                    rel: "noopener noreferrer",
                    "{url}"
                }
            }
            div { class: "credit-desc", "{description}" }
            pre { class: "credit-license", "{license}" }
        }
    }
}
