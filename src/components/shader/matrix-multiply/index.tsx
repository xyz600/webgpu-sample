import { useEffect, useRef } from "react";
import { type GPUResource, useGPUResource } from "../../WebGPUCanvas";
import matmulShader from "./shaders/matmul.wgsl";

type ProblemResource = {
	matrixSize: number;
	inputBuffer1: Float32Array;
	inputBuffer2: Float32Array;
	outputBuffer: Float32Array;
};

const createProblem = (matrixSize: number): ProblemResource => {
	const inputBuffer1 = new Float32Array(matrixSize * matrixSize);
	const inputBuffer2 = new Float32Array(matrixSize * matrixSize);
	for (let idx = 0; idx < inputBuffer1.length; idx += 1) {
		inputBuffer1[idx] = Math.random();
		inputBuffer2[idx] = Math.random();
	}
	const outputBuffer = new Float32Array(matrixSize * matrixSize);
	outputBuffer.fill(0.0);

	for (let i = 0; i < matrixSize; i += 1) {
		for (let j = 0; j < matrixSize; j += 1) {
			for (let k = 0; k < matrixSize; k += 1) {
				outputBuffer[i * matrixSize + j] +=
					inputBuffer1[i * matrixSize + k] * inputBuffer2[k * matrixSize + j];
			}
		}
	}
	return {
		matrixSize,
		inputBuffer1,
		inputBuffer2,
		outputBuffer,
	};
};

type ComputeResource = {
	shaderModule: string;
	pipeline: GPUComputePipeline;
	bindingGroups: GPUBindGroup;
	inputBuffer1: GPUBuffer;
	inputBuffer2: GPUBuffer;
	outputBuffer: GPUBuffer;
	resultBuffer: GPUBuffer;
};

const FLOAT32_BYTE_SIZE: number = 4;

const createComputeResource = (
	{ device }: GPUResource,
	problem: ProblemResource,
): ComputeResource => {
	const module = device.createShaderModule({
		label: "matrix multipulation module",
		code: matmulShader,
	});
	const matrixSize = 1024;
	const bufferSize = matrixSize * matrixSize * FLOAT32_BYTE_SIZE;

	const inputBuffer1 = device.createBuffer({
		label: "input buffer 1",
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(inputBuffer1, 0, problem.inputBuffer1);
	const inputBuffer2 = device.createBuffer({
		label: "input buffer 2",
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(inputBuffer2, 0, problem.inputBuffer2);
	const outputBuffer = device.createBuffer({
		label: "output buffer",
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const resultBuffer = device.createBuffer({
		label: "output buffer",
		size: bufferSize,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const pipeline = device.createComputePipeline({
		label: "matmul calculation",
		layout: "auto",
		compute: {
			module,
		},
	});

	const bindingGroups = device.createBindGroup({
		label: "binding groups",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer1 } },
			{ binding: 1, resource: { buffer: inputBuffer2 } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});

	return {
		shaderModule: matmulShader,
		pipeline,
		bindingGroups,
		inputBuffer1,
		inputBuffer2,
		outputBuffer,
		resultBuffer,
	};
};

const spawn = async (
	{ device }: GPUResource,
	computeResource: ComputeResource,
	problem: ProblemResource,
) => {
	const encoder = device.createCommandEncoder({
		label: "compute builtin encoder",
	});
	const pass = encoder.beginComputePass({ label: "compute builtin pass" });
	pass.setPipeline(computeResource.pipeline);
	pass.setBindGroup(0, computeResource.bindingGroups);
	const wgX = problem.matrixSize / 16;
	const wgY = problem.matrixSize / 16;
	pass.dispatchWorkgroups(wgX, wgY, 1);
	pass.end();
	const bufferSize =
		problem.matrixSize * problem.matrixSize * FLOAT32_BYTE_SIZE;
	encoder.copyBufferToBuffer(
		computeResource.outputBuffer,
		0,
		computeResource.resultBuffer,
		0,
		bufferSize,
	);
	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);
	await computeResource.resultBuffer.mapAsync(GPUMapMode.READ);
	const result = new Float32Array(
		computeResource.resultBuffer.getMappedRange(),
	);

	// 比較
	let diffSum = 0.0;
	for (let idx = 0; idx < problem.matrixSize * problem.matrixSize; idx += 1) {
		diffSum += Math.abs(result[idx] - problem.outputBuffer[idx]);
	}
	console.log(problem.outputBuffer.slice(0, 10));
	console.log(result.slice(0, 10));
	diffSum /= problem.matrixSize * problem.matrixSize;
	console.log(`average diff sum: ${diffSum}`);

	computeResource.resultBuffer.unmap();
};

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
		console.log("creating compute resource...");
		const computeResource = createComputeResource(deviceResource, problem);
		console.log("calculating gpu result...");
		spawn(deviceResource, computeResource, problem);
	}, [deviceResource]);

	return null;
};
