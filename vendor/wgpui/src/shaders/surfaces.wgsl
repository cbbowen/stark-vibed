struct Globals {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    // STARK PATCH: 1 when the swapchain is linear (scRGB). An embedder's surface
    // needs no decode — its texels are linear already — only the reference white.
    linear_output: u32,
    // STARK PATCH: scRGB units per SDR white, which is not 1 — see
    // `WgpuRenderer::sdr_white_scale`. What puts this surface on the same white as
    // the chrome around it.
    sdr_white_scale: f32,
}

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
}

struct SurfaceVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) clip_distances: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
// STARK PATCH: an array indexed by the instance, where it was a single uniform.
// One buffer written per frame holds every surface the scene paints, so two
// surfaces no longer read one another's bounds — see `WgpuRenderer::draw`.
@group(1) @binding(0) var<storage, read> b_surfaces: array<SurfaceParams>;
@group(1) @binding(1) var t_surface: texture_2d<f32>;
@group(1) @binding(2) var s_surface: sampler;

fn to_device_position(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

@vertex
fn vs_surface(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> SurfaceVarying {
    let params = b_surfaces[instance_id];
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let position = unit_vertex * params.bounds.size + params.bounds.origin;

    let clip_origin = params.content_mask.origin;
    let clip_size = params.content_mask.size;
    let tl = position - clip_origin;
    let br = clip_origin + clip_size - position;

    var out: SurfaceVarying;
    out.position = to_device_position(position);
    out.tex_coord = unit_vertex;
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);
    return out;
}

@fragment
fn fs_surface(input: SurfaceVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let color = textureSample(t_surface, s_surface, input.tex_coord);
    let alpha = color.a;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    // STARK PATCH: an embedder writes linear light with 1.0 at SDR white (that is
    // what `Window::surface_color_space` asks of it), and scRGB puts SDR white at
    // `sdr_white_scale`. Scaled, not decoded: these texels are not sRGB-encoded.
    let scale = select(1.0, globals.sdr_white_scale, globals.linear_output != 0u);
    return vec4<f32>(color.rgb * multiplier * scale, alpha);
}
