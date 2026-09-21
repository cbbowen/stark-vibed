//! The engine's integration tests, as one binary: a test crate per file would each
//! recompile `common` and link wgpu again. Cargo discovers only this file, so a module
//! missing from the list below is never compiled.

mod common;

mod assets;
mod assist;
mod blend;
mod collab;
mod color_dynamics;
mod commute;
mod commute_pairs;
mod composite;
mod corpus;
mod dynamics;
mod erase;
mod export;
mod fill;
mod filter;
mod footprint;
mod golden;
mod gradient;
mod groups;
mod guides;
mod hdr;
mod hover;
mod layers;
mod liquify;
mod matte;
mod merge;
mod minify;
mod modulation;
mod opacity;
mod peer_state;
mod pick;
mod place;
mod reference;
mod replay;
mod reset;
mod retention;
mod save_load;
mod seam;
mod selection;
mod sharing;
mod stroke;
mod tile_pool;
mod tooth;
mod transform;
mod translate;
mod view;
mod wide;
