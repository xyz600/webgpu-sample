@group(0) @binding(0) var<storage, read> in1: array<f32>;
@group(0) @binding(1) var<storage, read> in2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const WorkgroupSize: u32 = 8;
const WorkgroupSize2: u32 = WorkgroupSize * WorkgroupSize;
const SubMatrixSize: u32 = 32;
const SubMatrixSize2: u32 = SubMatrixSize * SubMatrixSize;
const UnitSize: u32 = SubMatrixSize / WorkgroupSize;
const UnitSize2: u32 = UnitSize * UnitSize;

var<workgroup> cache_in1: array<f32, SubMatrixSize2>;
var<workgroup> cache_in2: array<f32, SubMatrixSize2>;
var<workgroup> cache_out: array<f32, SubMatrixSize2>;

override matrixSize: u32;

@compute @workgroup_size(WorkgroupSize, WorkgroupSize, 1)
fn matmul(
    @builtin(workgroup_id) wid: vec3u, 
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(local_invocation_index) lindex: u32,
) {
    // buffer for register blocking
    var l_out = array<f32, UnitSize2>();
    for (var i: u32 = 0; i < UnitSize2; i += 1) {
        l_out[i] = 0.0;
    }
    var l_in1 = array<f32, UnitSize2>();
    var l_in2 = array<f32, UnitSize2>();

    for (var k: u32 = 0; k < matrixSize; k += SubMatrixSize) {
        // 1. cache blocking
        {
            let ly = lid.y * UnitSize;
            let lx = lid.x * UnitSize;

            let gy1 = wid.y * SubMatrixSize + ly;
            let gy2 = k + ly; 

            let gx1 = k + lx; 
            let gx2 = wid.x * SubMatrixSize + lx;

            // fixme: cache access pattern
            for (var lly: u32 = 0; lly < UnitSize; lly += 1) {
                for (var llx: u32 = 0; llx < UnitSize; llx += 1) {
                    cache_in1[(ly + lly) * SubMatrixSize + lx + llx] = in1[(gy1 + lly) * matrixSize + gx1 + llx];
                    cache_in2[(ly + lly) * SubMatrixSize + lx + llx] = in2[(gy2 + lly) * matrixSize + gx2 + llx];
                }
            }
        }
        workgroupBarrier();

        // 2. register blocking
        for (var kk: u32 = 0; kk < SubMatrixSize; kk += UnitSize) {
            let ly: u32 = lid.y * UnitSize;
            let lx: u32 = lid.x * UnitSize;

            // register blocking load
            for (var lly: u32 = 0; lly < UnitSize; lly += 1) {
                for (var llx: u32 = 0; llx < UnitSize; llx += 1) {
                    l_in1[lly * UnitSize + llx] = cache_in1[(ly + lly) * SubMatrixSize + kk + llx];
                    l_in2[lly * UnitSize + llx] = cache_in2[(kk + lly) * SubMatrixSize + lx + llx];
                }
            }

            for (var i: u32 = 0; i < UnitSize; i += 1) {
                for (var kkk: u32 = 0; kkk < UnitSize; kkk += 1) {
                    for (var j: u32 = 0; j < UnitSize; j += 1) {
                        l_out[i * UnitSize + j] += l_in1[i * UnitSize + kkk] * l_in2[kkk * UnitSize + j];
                    }
                }
            }
        }
        workgroupBarrier();
    }

    // 3. store global buffer
    {
        let gy = wid.y * SubMatrixSize + lid.y * UnitSize;
        let gx = wid.x * SubMatrixSize + lid.x * UnitSize;

        // fixme: cache access pattern
        for (var lly: u32 = 0; lly < UnitSize; lly += 1) {
            for (var llx: u32 = 0; llx < UnitSize; llx += 1) {
                out[(gy + lly) * matrixSize + gx + llx] = l_out[lly * UnitSize + llx];
            }
        }
    }
}
