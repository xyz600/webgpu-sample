@group(0) @binding(0) var<storage, read> in1: array<f32>;
@group(0) @binding(1) var<storage, read> in2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> debug_dump: array<f32>;

const WorkgroupSize: u32 = 8;
const WorkgroupSize2: u32 = WorkgroupSize * WorkgroupSize;
const SubMatrixSize: u32 = 32;
const SubMatrixSize2: u32 = SubMatrixSize * SubMatrixSize;
const UnitSize: u32 = SubMatrixSize / WorkgroupSize;
const UnitSize2: u32 = UnitSize * UnitSize;

var<workgroup> cache_in1: array<f32, SubMatrixSize2>;
var<workgroup> cache_in2: array<f32, SubMatrixSize2>;

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
            let lx: u32 = lindex % SubMatrixSize;
            let b_ly: u32 = lindex / SubMatrixSize;
            let y_step: u32 = WorkgroupSize2 / SubMatrixSize;

            let gx1: u32 = k * SubMatrixSize + lx;
            for (var ly: u32 = b_ly; ly < SubMatrixSize; ly += y_step) {
                let gy1: u32 = wid.y * SubMatrixSize + ly;
                cache_in1[ly * SubMatrixSize + lx] = in1[gy1 * matrixSize + gx1];
            }

            let gx2: u32 = wid.x * SubMatrixSize + lx;
            for (var ly: u32 = b_ly; ly < SubMatrixSize; ly += y_step) {
                let gy2: u32 = k * SubMatrixSize + ly;
                cache_in2[ly * SubMatrixSize + lx] = in2[gy2 * matrixSize + gx2];
            }
        }
        workgroupBarrier();

        if (k > 0) {
            let ly: u32 = lid.y * UnitSize;
            let lx: u32 = lid.x * UnitSize;

            let gy = wid.y * SubMatrixSize + ly;
            let gx = wid.x * SubMatrixSize + lx;

            for (var lly: u32 = 0; lly < UnitSize; lly += 1) {
                for (var llx: u32 = 0; llx < UnitSize; llx += 1) {
                    debug_dump[(gy + lly) * matrixSize + gx + llx] = cache_in1[(ly + lly) * SubMatrixSize + lx + llx];
                }
            }

            break;
        }

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
        if (k == 0) {
            for (var i: u32 = 0; i < UnitSize; i += 1) {
                for (var j: u32 = 0; j < UnitSize; j += 1) {
                    l_out[i * UnitSize + j] = 0;
                }
            }            
        }
        if (k > 0) {
            break;
        }
    }

    // 3. store global result
    {
        let ly = lid.y * UnitSize;
        let lx = lid.x * UnitSize;

        let gy = wid.y * SubMatrixSize + ly;
        let gx = wid.x * SubMatrixSize + lx;

        // fixme: cache access pattern
        for (var lly: u32 = 0; lly < UnitSize; lly += 1) {
            for (var llx: u32 = 0; llx < UnitSize; llx += 1) {
                out[(gy + lly) * matrixSize + gx + llx] = l_out[lly * UnitSize + llx];
            }
        }
    }
}
