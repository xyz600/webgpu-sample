export const getGPUDevice = async (): Promise<GPUDevice> => {
    if (navigator.gpu === undefined) {
        throw new Error("webgpu not supported");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
        throw new Error("cannot get gpu adapter");
    }
    const canTimestamp = adapter.features.has("timestamp-query");
    if (!canTimestamp) {
        console.warn("timestamp-query is not supported");
    }
    const device = await adapter.requestDevice({
        requiredFeatures: ["timestamp-query"],
    });
    return device;
};
