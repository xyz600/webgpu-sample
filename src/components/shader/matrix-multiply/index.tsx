import matmulShader from "./shaders/matmul.wgsl";
import {
	CPUMatmulClient,
	GPUMatmulClient,
	checkDifference,
	createProblem,
} from "./client";
import { Button, HStack } from "@chakra-ui/react";
import { getGPUDevice } from "../../../utils/device";
import { useMemo, useState } from "react";

export const MatrixMultipulation = () => {
	const matrixSize = 1024;
	const problem = useMemo(() => createProblem(matrixSize), []);

	const [cpuResult, setCpuResult] = useState<Float32Array | null>();
	const [gpuResult, setGpuResult] = useState<Float32Array | null>();

	const triggerCPU = () => {
		const cpuClient = new CPUMatmulClient(problem);
		const cpuStart = performance.now();
		console.log("creating CPU calculation...");
		const cpuResult = cpuClient.calculate();
		const cpuStop = performance.now();
		const elapsed = cpuStop - cpuStart;
		console.log(`CPU calculation done in ${elapsed} ms`);
		console.log(cpuResult.slice(0, 10));
		setCpuResult(cpuResult);
	};

	const triggerGPU = async () => {
		const device = await getGPUDevice();
		console.log("creating problem...");

		console.log("creating GPU calculation...");
		const gpuStart = performance.now();
		const gpuClient = new GPUMatmulClient(device, matmulShader, problem);
		const gpuResult = await gpuClient.calculate();
		const gpuStop = performance.now();
		const elapsed = gpuStop - gpuStart;
		console.log(`GPU calculation done in ${elapsed} ms`);
		gpuClient.destroy();
		console.log(gpuResult.slice(0, 10));
		setGpuResult(gpuResult);
	};

	const averageDiff = useMemo(
		() => checkDifference(cpuResult, gpuResult),
		[cpuResult, gpuResult],
	);
	if (typeof averageDiff === "undefined") {
		console.log("both calculation are not done yet.");
	} else {
		console.log("average diff: ", averageDiff);
	}

	return (
		<HStack>
			<Button onClick={triggerGPU}> GPU 実行</Button>
			<Button onClick={triggerCPU}> CPU 実行</Button>
		</HStack>
	);
};
