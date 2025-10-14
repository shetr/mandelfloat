
@group(0) @binding(0) var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let color = vec4<f32>(1.0);

    textureStore(output, location, color);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let textureDimensions = textureDimensions(output);
    let ratio = f32(textureDimensions.x) / f32(textureDimensions.y);

    var i = 0;
    var z = vec2<f32>(0.0);
    var z2 = vec2<f32>(0.0);
    let c = (vec2<f32>(location) / vec2<f32>(textureDimensions) * 2.0 - 1.0) * vec2<f32>(ratio, 1.0);
    while (i <= 100 && z2.x + z2.y <= 4.0)
    {
        z = vec2<f32>(z2.x - z2.y, 2.0*z.x*z.y) + c;
        z2 = z * z;
        i++;
    }

    let color = vec4<f32>(f32(i) / 100.0);

    textureStore(output, location, color);
}