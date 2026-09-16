//! A bind group layout read off the shader (§6.10): what the entry points sharing one
//! layout reach in one `@group`, and the `wgpu` entries for it.
//!
//! The host had been writing that list by hand — a slot per binding, each restating
//! whether it is a uniform and how wide, a sampler, a texture of a scalar, a storage
//! texture of a format and an access mode. All of that is in the declaration.
//!
//! **Three things are left to the host**, because no declaration states them: which
//! entry points share the layout, which uniforms are bound at a dynamic offset, and —
//! through the first — what the layout is *for*.

use std::collections::BTreeMap;

use crate::{BindKind, Binding, EntryPoint};

/// One binding a set of entry points reaches in one `@group`, folded over the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reach {
    /// The shader's declaration of the slot.
    pub decl: Binding,
    /// Whether **any** of them reads it through a sampler, which is what makes its
    /// texture filterable. `|`, not the last one's answer: one layout serves every
    /// sharer, so a texture one of them samples has to be declared filterable even
    /// where the rest load it.
    pub sampled: bool,
    /// The stages that reach it — the union, which is exactly the visibility its
    /// layout entry needs and no more.
    pub visibility: wgpu::ShaderStages,
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
pub fn reached(eps: &[EntryPoint], anchor: Binding) -> Vec<Reach> {
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

/// The bind group layout entries for `anchor`'s `@group`, as `eps` reach it.
///
/// Pass every entry point that shares the layout: the entries are their union, so one
/// left out is a slot missing, an over-narrow visibility, or a texture declared
/// unfilterable that something samples. Pass **one** where the layout has to be exact
/// — `wgpu` merges a whole bind group into each dispatch's usage scope, so a compute
/// kernel that writes a storage texture another kernel samples needs a layout of its
/// own rather than a union with it.
///
/// `dynamic` names the uniforms bound as one slot of a larger buffer
/// (`gpu::uniforms`), which is the one thing about a `var<uniform>` the WGSL cannot
/// say: it is identical either way.
///
/// # Panics
/// If the group is empty — the anchor names a group these entry points do not reach —
/// or if `dynamic` names something that is not a uniform of it.
pub fn layout_entries(
    eps: &[EntryPoint],
    anchor: Binding,
    dynamic: &[Binding],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let reach = reached(eps, anchor);
    assert!(
        !reach.is_empty(),
        "`{}.wesl`'s `{}` anchors @group({}), which none of these entry points reach",
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
        // Filtering either way: the flag describes the image side of the pair, and a
        // sampler nothing samples with is bound but never read.
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
        let entries = layout_entries(
            &[c.vs_main, c.fs_main, m.vs_main, m.fs_main],
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
        let entries = layout_entries(&[c.vs_main, c.fs_main], cd::TILE_COLOR, &[]);
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
            layout_entries(&[c.vs_main, c.fs_main], cd::TILE_COLOR, &[])
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

    /// An anchor naming a group nothing here reaches is a mistake in the call, not an
    /// empty layout.
    #[test]
    #[should_panic(expected = "which none of these entry points reach")]
    fn an_anchor_outside_what_the_entry_points_reach_is_refused() {
        let c = composite(Resid::Without);
        layout_entries(&[c.vs_main], cd::TILE_COLOR, &[]);
    }

    /// A dynamic offset asked of a slot the group does not hold.
    #[test]
    #[should_panic(expected = "has no such uniform here")]
    fn a_dynamic_offset_on_a_slot_of_another_group_is_refused() {
        let c = composite(Resid::Without);
        layout_entries(&[c.vs_main, c.fs_main], cd::TILE_COLOR, &[vd::VIEW]);
    }
}
