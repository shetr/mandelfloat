
@group(0) @binding(0) var<storage, read> input: array<IterationResult>;

@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2) var<uniform> config: VisualizeUniforms;

@group(0) @binding(3) var<uniform> shared_config: SharedUniforms;

struct VisualizeUniforms {
    min_color: vec4<f32>,
    max_color: vec4<f32>,
}

struct SharedUniforms {
    max_iterations: u32,
}

struct IterationResult {
    z: vec2<f32>,
    i: u32,
}

@compute @workgroup_size(8, 8, 1)
fn visualize(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let textureDimensions = textureDimensions(output);

    let input_index = invocation_id.x + textureDimensions.x * invocation_id.y;
    let i = input[input_index].i;

    let t = vec4<f32>(f32(i) / f32(shared_config.max_iterations));
    let color = (1.0 - t) * config.min_color + t * config.max_color;

    textureStore(output, location, color);
}