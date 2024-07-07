const FLOAT32_BYTE_SIZE: number = 4;

type ProblemResource = {
    matrixSize: number;
    inputBuffer1: Float32Array;
    inputBuffer2: Float32Array;
};

export const createProblem = (matrixSize: number): ProblemResource => {
    const inputBuffer1 = new Float32Array(matrixSize * matrixSize);
    const inputBuffer2 = new Float32Array(matrixSize * matrixSize);
    for (let idx = 0; idx < inputBuffer1.length; idx += 1) {
        inputBuffer1[idx] = Math.random();
        inputBuffer2[idx] = Math.random();
    }
    return {
        matrixSize,
        inputBuffer1,
        inputBuffer2,
    };
};

export const checkDifference = (a: Float32Array, b: Float32Array): number => {
    let diff = 0;
    console.log(a.length, b.length);
    for (let idx = 0; idx < a.length; idx += 1) {
        diff += Math.abs(a[idx] - b[idx]);
    }
    return diff / a.length;
};

export class CPUMatmulClient {
    outputBuffer: Float32Array;
    problem: ProblemResource;

    constructor(problem: ProblemResource) {
        const matrixSize = problem.matrixSize;
        this.outputBuffer = new Float32Array(matrixSize * matrixSize);
        this.problem = problem;
    }

    calculate() {
        this.outputBuffer.fill(0.0);
        const matrixSize = this.problem.matrixSize;

        for (let i = 0; i < matrixSize; i += 1) {
            for (let k = 0; k < matrixSize; k += 1) {
                for (let j = 0; j < matrixSize; j += 1) {
                    this.outputBuffer[i * matrixSize + j] +=
                        this.problem.inputBuffer1[i * matrixSize + k] * this.problem.inputBuffer2[k * matrixSize + j];
                }
            }
        }
        return this.outputBuffer;
    }

    dtor() {
    }
}

export class GPUMatmulClient {
    pipeline: GPUComputePipeline;
    bindingGroups: GPUBindGroup;
    inputBuffer1: GPUBuffer;
    inputBuffer2: GPUBuffer;
    outputBuffer: GPUBuffer;
    queryBuffer: GPUBuffer;
    queryReadBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
    device: GPUDevice;
    problem: ProblemResource;

    constructor(device: GPUDevice, matmulShader: string, problem: ProblemResource) {
        this.problem = problem;
        this.device = device;
        const module = device.createShaderModule({
            label: "matrix multipulation module",
            code: matmulShader,
        });
        const matrixSize = problem.matrixSize;
        const bufferSize = matrixSize * matrixSize * FLOAT32_BYTE_SIZE;

        this.inputBuffer1 = device.createBuffer({
            label: "input buffer 1",
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.inputBuffer1, 0, problem.inputBuffer1);
        this.inputBuffer2 = device.createBuffer({
            label: "input buffer 2",
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.inputBuffer2, 0, problem.inputBuffer2);
        this.outputBuffer = device.createBuffer({
            label: "output buffer",
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.resultBuffer = device.createBuffer({
            label: "result buffer",
            size: bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.queryBuffer = device.createBuffer({
            label: "query buffer for timestamp query",
            size: 8 * 2,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        this.queryReadBuffer = device.createBuffer({
            label: "query buffer for timestamp query",
            size: 8 * 2,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.pipeline = device.createComputePipeline({
            label: "matmul calculation",
            layout: "auto",
            compute: {
                module,
            },
        });
        this.bindingGroups = device.createBindGroup({
            label: "binding groups",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer1 } },
                { binding: 1, resource: { buffer: this.inputBuffer2 } },
                { binding: 2, resource: { buffer: this.outputBuffer } },
            ],
        });
    }

    async calculate(): Promise<Float32Array> {
        const encoder = this.device.createCommandEncoder({
            label: "compute builtin encoder",
        });

        const querySet = this.device.createQuerySet({
            label: "timestamp query set",
            type: "timestamp",
            count: 2,
        });

        encoder.writeTimestamp(querySet, 0);
        const pass = encoder.beginComputePass({ label: "compute builtin pass" });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindingGroups);
        const wgX = this.problem.matrixSize / 16;
        const wgY = this.problem.matrixSize / 16;
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
        const bufferSize =
            this.problem.matrixSize * this.problem.matrixSize * FLOAT32_BYTE_SIZE;
        encoder.writeTimestamp(querySet, 1);
        encoder.copyBufferToBuffer(
            this.outputBuffer,
            0,
            this.resultBuffer,
            0,
            bufferSize,
        );
        encoder.resolveQuerySet(querySet, 0, querySet.count, this.queryBuffer, 0);
        encoder.copyBufferToBuffer(
            this.queryBuffer, 0, this.queryReadBuffer, 0, 8 * 2
        );

        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        await this.resultBuffer.mapAsync(GPUMapMode.READ, 0, bufferSize);
        const result = new Float32Array(
            this.resultBuffer.getMappedRange()
        ).slice();
        this.resultBuffer.unmap();

        // バッファからデータを読み取る
        await this.queryReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.queryReadBuffer.getMappedRange();
        const timestamps = new BigUint64Array(arrayBuffer);

        // 実行時間を計算
        const executionTime = (timestamps[1] - timestamps[0]) / BigInt(1000); // ナノ秒をマイクロ秒に変換
        console.log(`GPUの実行時間: ${executionTime} [us]`);
        this.queryReadBuffer.unmap();

        return result;
    }

    destroy() {
        this.inputBuffer1.destroy();
        this.inputBuffer2.destroy();
        this.outputBuffer.destroy();
        this.resultBuffer.destroy();
    }
}
