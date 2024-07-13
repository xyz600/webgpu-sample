
const UINT64_BYTE_SIZE = 8;
const TIMER_COUNT: number = 2;

/**
 * 特定の関数の実行時間を計測するための簡易ライブラリ
 */
export class GPUTimeMeasure {
    queryBuffer: GPUBuffer;
    queryReadBuffer: GPUBuffer;
    querySet: GPUQuerySet;

    constructor(device: GPUDevice, name: string) {
        this.querySet = device.createQuerySet({
            label: `timestamp query set: name = ${name}`,
            type: "timestamp",
            count: 2,
        });

        this.queryBuffer = device.createBuffer({
            label: `timestamp query buffer: name = ${name}`,
            size: UINT64_BYTE_SIZE * TIMER_COUNT,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        this.queryReadBuffer = device.createBuffer({
            label: `timestamp read buffer: name = ${name}`,
            size: UINT64_BYTE_SIZE * TIMER_COUNT,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    setup(encoder: GPUCommandEncoder, func: (encoder: GPUCommandEncoder) => void) {
        // @ts-expect-error only chrome is supported
        encoder.writeTimestamp(this.querySet, 0);
        func(encoder);
        // @ts-expect-error only chrome is supported
        encoder.writeTimestamp(this.querySet, 1);
        encoder.resolveQuerySet(this.querySet, 0, TIMER_COUNT, this.queryBuffer, 0);
        encoder.copyBufferToBuffer(
            this.queryBuffer, 0, this.queryReadBuffer, 0, UINT64_BYTE_SIZE * TIMER_COUNT
        );
    }

    async elapsedTimeUs(): Promise<bigint> {
        await this.queryReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.queryReadBuffer.getMappedRange();
        const timestamps = new BigUint64Array(arrayBuffer);

        const executionTime = BigInt(timestamps[1] - timestamps[0]) / BigInt(1000); // ナノ秒をマイクロ秒に変換
        this.queryReadBuffer.unmap();
        return executionTime;
    }

    destroy() {
        this.queryBuffer.destroy();
        this.queryReadBuffer.destroy();
    }
}