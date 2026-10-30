@group(0)
@binding(0)
var canvas: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = textureLoad(canvas, vec2<i32>(position.xy), 0);
    let background = vec3<f32>(0.004, 0.006, 0.014);
    return vec4<f32>(mix(background, texel.rgb, texel.a), 1.0);
}
