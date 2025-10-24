
@group(0) @binding(0) var z_x: texture_storage_2d<r32float, read_write>;

@group(0) @binding(1) var z_y: texture_storage_2d<r32float, read_write>;

@group(0) @binding(2) var iteration: texture_storage_2d<r32uint, read_write>;

@group(0) @binding(3) var<uniform> config: IterationsUniforms;

@group(0) @binding(4) var<uniform> shared_config: SharedUniforms;

struct IterationsUniforms {
    transform: mat3x3<f32>,
}

struct SharedUniforms {
    max_iterations: u32,
}

@compute @workgroup_size(8, 8, 1)
fn iterate(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let dimensions = textureDimensions(iteration);

    let ratio = f32(dimensions.x) / f32(dimensions.y);

    let normalized_location = (vec2<f32>(location) / vec2<f32>(dimensions) * 2.0 - 1.0) * vec2<f32>(ratio, -1.0);
    let positon = (config.transform * vec3<f32>(normalized_location, 1.0)).xy;

    var i = 0u;
    var z = vec2<f32>(0.0);
    var z2 = vec2<f32>(0.0);
    let c = positon;
    while (i <= shared_config.max_iterations && z2.x + z2.y <= 4.0)
    {
        z = vec2<f32>(z2.x - z2.y, 2.0*z.x*z.y) + c;
        z2 = z * z;
        i++;
    }

    textureStore(iteration, location, vec4<u32>(i));
    textureStore(z_x, location, vec4<f32>(z.x));
    textureStore(z_y, location, vec4<f32>(z.y));
}