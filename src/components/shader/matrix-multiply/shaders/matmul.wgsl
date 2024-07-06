@group(0) @binding(0) var<storage, read> in1: array<f32>;
@group(0) @binding(1) var<storage, read> in2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u
) {
    const SIZE: u32 = 1024;
    let xi: u32 = wid.x * 16 + lid.x;
    let yi: u32 = wid.y * 16 + lid.y;
    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < SIZE; k += 1) {
        sum += in1[xi * SIZE + k] * in2[k * SIZE + yi];
    }
    out[xi * SIZE + yi] = sum;
}