@group(0) @binding(0) var<storage, read> in1: array<f32>;
@group(0) @binding(1) var<storage, read> in2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

override matrixSize: u32;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u
) {
    let xi: u32 = wid.x * 16 + lid.x;
    let yi: u32 = wid.y * 16 + lid.y;
    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < matrixSize; k += 1) {
        sum += in1[xi * matrixSize + k] * in2[k * matrixSize + yi];
    }
    out[xi * matrixSize + yi] = sum;
}