const FLOAT32_BYTE_SIZE = 4;

/**
 * デバッグ用に、GPUBufferの中身を都度確認するための簡易 client
 */
export class DebugDumper {
    buffer: GPUBuffer;
    readBuffer: GPUBuffer;
    bufferSize: number;

    constructor(device: GPUDevice, name: string, bufferSize: number) {
        this.bufferSize = bufferSize;
        this.buffer = device.createBuffer({
            label: `debug dumper buffer: name = ${name}`,
            size: FLOAT32_BYTE_SIZE * bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.readBuffer = device.createBuffer({
            label: `debug dumper read buffer: name = ${name}`,
            size: FLOAT32_BYTE_SIZE * bufferSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    setup(encoder: GPUCommandEncoder, func: (encoder: GPUCommandEncoder) => void) {
        func(encoder);
        encoder.copyBufferToBuffer(
            this.buffer, 0, this.readBuffer, 0, FLOAT32_BYTE_SIZE * this.bufferSize
        );
    }

    getBindingGroup(index: number): GPUBindGroupEntry {
        return { binding: index, resource: { buffer: this.buffer } };
    }

    async read(): Promise<Float32Array> {
        await this.readBuffer.mapAsync(GPUMapMode.READ, 0, FLOAT32_BYTE_SIZE * this.bufferSize);
        const arrayBuffer = this.readBuffer.getMappedRange();
        const buffer = new Float32Array(arrayBuffer).slice();
        this.readBuffer.unmap();
        return buffer;
    }

    destroy() {
        this.buffer.destroy();
        this.readBuffer.destroy();
    }
}