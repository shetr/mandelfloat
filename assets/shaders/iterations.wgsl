
@group(0) @binding(0) var<storage, read> input: array<IterationResult>;

@group(0) @binding(1) var<storage, read_write> output: array<IterationResult>;

@group(0) @binding(2) var<uniform> config: IterationsUniforms;

@group(0) @binding(3) var<uniform> shared_config: SharedUniforms;

struct IterationsUniforms {
    transform: mat3x3<f32>,
    dimensions: vec2<u32>
}

struct SharedUniforms {
    max_iterations: u32,
}

struct IterationResult {
    z: vec2<f32>,
    i: u32,
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = invocation_id.xy;
    let dimensions = config.dimensions;
    let storage_index = location.x + dimensions.x * location.y;

    output[storage_index].i = 0u;
    output[storage_index].z = vec2<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = invocation_id.xy;
    let dimensions = config.dimensions;
    let storage_index = location.x + dimensions.x * location.y;

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

    output[storage_index].i = i;
    output[storage_index].z = z;
}