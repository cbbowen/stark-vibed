//! A bind group layout read off the shader (§6.10): what the pipelines over one layout
//! reach in one `@group`, and the `wgpu` entries for it.
//!
//! **Three things are left to the host**, because no declaration states them: which
//! pipelines share the layout ([`layout_of`] against [`layout_shared_by`]), which
//! uniforms are bound at a dynamic offset, and — through the first — what the layout
//! is *for*.

use std::collections::BTreeMap;

use crate::{BindKind, Binding, EntryPoint};

/// One binding a set of entry points reaches in one `@group`, folded over the set.
#[derive(Clone, Copy)]
struct Reach {
    /// The shader's declaration of the slot.
    decl: Binding,
    /// Whether **any** of them reads it through a sampler, which is what makes its
    /// texture filterable. `|`, not the last one's answer: one layout serves every
    /// sharer, so a texture one of them samples has to be declared filterable even
    /// where the rest load it.
    sampled: bool,
    /// The stages that reach it — the union, which is exactly the visibility its
    /// layout entry needs and no more.
    visibility: wgpu::ShaderStages,
}

/// Every binding `eps` reach in `anchor`'s `@group`, ascending by index.
///
/// `anchor` is any declaration in the group, and naming one is how a caller says which
/// group it means: a group *number* on the host is a transcription of what the WESL
/// already states, and one that a regrouping would leave silently wrong.
///
/// A slot the variant does not have — `@if(resid)`, or a module only one colour space
/// links — is simply absent from that variant's entry points, so presence needs no
/// argument here (§6.7).
///
/// # Panics
/// If two declarations claim one `@group`/`@binding`. Within one artifact the
/// generator refuses that; across two — `composite.wesl` and `matte.wesl` share a
/// layout — this is what does.
fn reached(eps: &[EntryPoint], anchor: Binding) -> Vec<Reach> {
    let mut out: BTreeMap<u32, Reach> = BTreeMap::new();
    for ep in eps {
        for used in ep.uses.iter().filter(|u| u.decl.group == anchor.group) {
            let at = out.entry(used.decl.index).or_insert(Reach {
                decl: used.decl,
                sampled: false,
                visibility: wgpu::ShaderStages::NONE,
            });
            assert!(
                at.decl == used.decl,
                "@group({}) @binding({}) is `{}.wesl`'s `{}` in one of these entry points \
                 and `{}.wesl`'s `{}` in another",
                used.decl.group,
                used.decl.index,
                at.decl.module,
                at.decl.name,
                used.decl.module,
                used.decl.name,
            );
            at.sampled |= used.sampled;
            at.visibility |= ep.stage;
        }
    }
    out.into_values().collect()
}

/// The stages of **one** pipeline, which is the unit a layout is asked for here — so
/// neither spelling below takes a loose bag of stages, in which a dropped `vs_` would
/// be an over-narrow visibility nothing but a device reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stages {
    /// A compute pipeline's kernel.
    Compute(EntryPoint),
    /// A render pipeline's two stages, which share a layout because one pipeline has
    /// one.
    Render(EntryPoint, EntryPoint),
}

impl Stages {
    /// Append this pipeline's entry points to `out`.
    fn push_to(self, out: &mut Vec<EntryPoint>) {
        match self {
            Self::Compute(k) => out.push(k),
            Self::Render(vs, fs) => out.extend([vs, fs]),
        }
    }
}

/// The bind group layout entries for `anchor`'s `@group`, **exactly** as one pipeline
/// reaches it.
///
/// The form a compute dispatch needs — see [`layout_shared_by`] for why, and for
/// `dynamic` and what is checked.
pub fn layout_of(
    stages: Stages,
    anchor: Binding,
    dynamic: &[Binding],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    layout_shared_by(&[stages], anchor, dynamic)
}

/// The bind group layout entries for `anchor`'s `@group`, as the **union** of what the
/// pipelines sharing it reach.
///
/// Pass every pipeline that shares the layout: one left out is a slot missing, an
/// over-narrow visibility, or a texture declared unfilterable that something samples.
///
/// **Sharing is the caller's claim, and a union cannot check it.** The entries widen to
/// cover every sharer, and `wgpu` merges a whole bind group into each dispatch's usage
/// scope — so two kernels that disagree about a texture, sampled in one and
/// storage-written in the next, come out of here as one valid-looking layout and fail
/// only at dispatch. Those need [`layout_of`] each.
///
/// `dynamic` names the uniforms bound as one slot of a larger buffer
/// (`gpu::uniforms`), which is the one thing about a `var<uniform>` the WGSL cannot
/// say: it is identical either way. Everything named must be such a uniform of this
/// group, which is checked — but a uniform *left out* of it is not, and comes out
/// whole-bound to fail at the first `set_bind_group` that passes an offset.
///
/// # Panics
/// If the group is empty — the anchor names a group these pipelines do not reach — or
/// if `dynamic` names something that is not a uniform of it.
pub fn layout_shared_by(
    pipelines: &[Stages],
    anchor: Binding,
    dynamic: &[Binding],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut eps = Vec::with_capacity(2 * pipelines.len());
    for p in pipelines {
        p.push_to(&mut eps);
    }
    entries(&eps, anchor, dynamic)
}

/// [`layout_shared_by`] for a group these pipelines may reach nothing of: the entries,
/// or **none**.
///
/// The colour-space split is what wants it (§6.7). A pigment document's blend, filter
/// and media passes each own a `@group(1)` — the inverse LUT and the residual — and a
/// colorimetric one declares nothing there. An empty answer is a real layout rather
/// than a missing one: a pipeline layout is positional, so the alternative is a hole
/// in it, and a hole is what the web backend is least happy with.
///
/// No `dynamic`: a group that may be empty is no place for a uniform bound at an
/// offset, and the check that would name one has nothing to check against.
pub fn layout_if_reached(pipelines: &[Stages], anchor: Binding) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut eps = Vec::with_capacity(2 * pipelines.len());
    for p in pipelines {
        p.push_to(&mut eps);
    }
    reached(&eps, anchor)
        .iter()
        .map(|r| entry(r, &[]))
        .collect()
}

/// The entries themselves, over the stages both spellings above flatten to.
fn entries(
    eps: &[EntryPoint],
    anchor: Binding,
    dynamic: &[Binding],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let reach = reached(eps, anchor);
    assert!(
        !reach.is_empty(),
        "`{}.wesl`'s `{}` anchors @group({}), which none of these pipelines reach",
        anchor.module,
        anchor.name,
        anchor.group,
    );
    for d in dynamic {
        assert!(
            reach
                .iter()
                .any(|r| r.decl == *d && matches!(r.decl.kind, BindKind::Uniform { .. })),
            "`{}.wesl`'s `{}` is bound at a dynamic offset, but @group({}) has no such \
             uniform here",
            d.module,
            d.name,
            anchor.group,
        );
    }
    reach.iter().map(|r| entry(r, dynamic)).collect()
}

/// One entry, from the declaration plus the two things folded over the set.
fn entry(r: &Reach, dynamic: &[Binding]) -> wgpu::BindGroupLayoutEntry {
    let ty = match r.decl.kind {
        // `min_binding_size` is the declared struct's own WGSL size — free validation
        // against a write shorter than what the shader reads.
        BindKind::Uniform { min_size } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic.contains(&r.decl),
            min_binding_size: wgpu::BufferSize::new(min_size),
        },
        // `Filtering` accepts any non-comparison sampler, and the generator refuses
        // `sampler_comparison` outright — so there is no declaration this can be wrong
        // for.
        BindKind::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        BindKind::Texture { dim, sample } => wgpu::BindingType::Texture {
            sample_type: sample.of(r.sampled),
            view_dimension: dim,
            multisampled: false,
        },
        BindKind::Storage {
            dim,
            format,
            access,
        } => wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension: dim,
        },
    };
    wgpu::BindGroupLayoutEntry {
        binding: r.decl.index,
        visibility: r.visibility,
        ty,
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mirror::composite::decl as cd;
    use crate::mirror::view::decl as vd;
    use crate::{Resid, composite, matte};

    /// Pass A's view group: the two stages split between the two slots, the uniform
    /// slotted, and the sampler filtering — every field of it derived.
    ///
    /// The layout is the composite pipeline's *and* the matte's, so both shaders'
    /// entry points are folded in; `view.wesl` declares the group and neither shader
    /// declares a slot of its own there.
    #[test]
    fn the_view_group_is_the_two_stages_split_between_its_two_slots() {
        let (c, m) = (composite(Resid::Without), matte(Resid::Without));
        let entries = layout_shared_by(
            &[
                Stages::Render(c.vs_main, c.fs_main),
                Stages::Render(m.vs_main, m.fs_main),
            ],
            vd::VIEW,
            &[vd::VIEW],
        );
        assert_eq!(
            entries,
            [
                wgpu::BindGroupLayoutEntry {
                    binding: vd::VIEW.index,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(match vd::VIEW.kind {
                            BindKind::Uniform { min_size } => min_size,
                            _ => unreachable!("`view.wesl`'s `view` is a uniform"),
                        }),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: vd::SAMP.index,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        );
    }

    /// Pass A's tile group: three sampled textures, so all three are filterable.
    #[test]
    fn a_sampled_tile_texture_is_declared_filterable() {
        let c = composite(Resid::Without);
        let entries = layout_of(Stages::Render(c.vs_main, c.fs_main), cd::TILE_COLOR, &[]);
        assert!(
            entries.iter().all(|e| matches!(
                e.ty,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    ..
                }
            )),
            "pass A samples its tiles through the view's sampler: {entries:?}",
        );
    }

    /// A slot only the residual variant declares is in that variant's layout and in no
    /// other — without the host restating the gate (§6.7).
    #[test]
    fn the_residual_channel_is_in_the_residual_variants_layout_alone() {
        let has_resid = |r| {
            let c = composite(r);
            layout_of(Stages::Render(c.vs_main, c.fs_main), cd::TILE_COLOR, &[])
                .iter()
                .any(|e| e.binding == cd::TILE_RESID.index)
        };
        const _: () = assert!(
            cd::TILE_RESID.resid,
            "the fixture no longer stands for an `@if(resid)` declaration",
        );
        assert!(!has_resid(Resid::Without), "a space with no residual");
        #[cfg(feature = "mixbox")]
        assert!(has_resid(Resid::With), "a pigment space");
    }

    /// Every entry point of a record names that record, which is what lets a host
    /// refuse a module built from one shader under another's entry points.
    #[test]
    fn an_entry_point_names_the_artifact_that_declares_it() {
        let c = composite(Resid::Without);
        for ep in c.entries {
            assert_eq!(ep.artifact, "composite", "`{}`", ep.name);
        }
        // The pair the sweep's two builds would confuse, were the name not carried:
        // `fs_levels` is not `@if(ceiling)`-gated, so every other field agrees.
        let (plain, lane) = (
            crate::stamp(Resid::Without, crate::Lane::Plain).fs_levels,
            crate::stamp(Resid::Without, crate::Lane::Ceiling).fs_levels,
        );
        assert_ne!(plain, lane, "the two builds' `fs_levels` are one value");
        assert_eq!((plain.name, lane.name), ("fs_levels", "fs_levels"));
    }

    /// An anchor naming a group nothing here reaches is a mistake in the call, not an
    /// empty layout.
    #[test]
    #[should_panic(expected = "which none of these pipelines reach")]
    fn an_anchor_outside_what_the_pipelines_reach_is_refused() {
        let c = composite(Resid::Without);
        layout_of(Stages::Compute(c.vs_main), cd::TILE_COLOR, &[]);
    }

    /// A dynamic offset asked of a slot the group does not hold.
    #[test]
    #[should_panic(expected = "has no such uniform here")]
    fn a_dynamic_offset_on_a_slot_of_another_group_is_refused() {
        let c = composite(Resid::Without);
        layout_of(
            Stages::Render(c.vs_main, c.fs_main),
            cd::TILE_COLOR,
            &[vd::VIEW],
        );
    }
}
