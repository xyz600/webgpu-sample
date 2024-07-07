import { useEffect, useRef } from "react";
import { useGPUResource } from "../../WebGPUCanvas";
import matmulShader from "./shaders/matmul.wgsl";
import {
	CPUMatmulClient,
	GPUMatmulClient,
	checkDifference,
	createProblem,
} from "./client";

export const MatrixMultipulation = () => {
	const deviceResource = useGPUResource();
	const matrixSize = 1024;

	const refSubmitted = useRef<boolean>(false);

	useEffect(() => {
		if (refSubmitted.current) {
			return;
		}
		refSubmitted.current = true;
		console.log("creating problem...");
		const problem = createProblem(matrixSize);

		console.log("creating CPU calculation...");
		const cpuClient = new CPUMatmulClient(problem);
		const cpuStart = performance.now();
		const cpuResult = cpuClient.calculate();
		const cpuStop = performance.now();
		const elapsed = cpuStop - cpuStart;
		console.log(`CPU calculation done in ${elapsed} ms`);

		console.log("creating GPU calculation...");
		(async () => {
			const gpuStart = performance.now();
			const gpuClient = new GPUMatmulClient(
				deviceResource.device,
				matmulShader,
				problem,
			);
			const gpuResult = await gpuClient.calculate();
			const gpuStop = performance.now();
			const elapsed = gpuStop - gpuStart;
			console.log(`GPU calculation done in ${elapsed} ms`);
			gpuClient.destroy();

			const averageDiff = checkDifference(cpuResult, gpuResult);
			console.log("average diff: ", averageDiff);
		})();
	}, [deviceResource]);

	return null;
};
