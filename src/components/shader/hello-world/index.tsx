import { useMemo } from "react";
import { useFrame } from "../../../hooks/useFrame";
import { type GPUResource, useGPUResource } from "../../WebGPUCanvas";
import fragWGSL from "./shaders/red.flag.wgsl";
import triangleVertWGSL from "./shaders/triangle.vert.wgsl";

const createUpdater = (
	{ device, context }: GPUResource,
	pipeline: GPURenderPipeline,
) => {
	const update = () => {
		const commandEncoder = device.createCommandEncoder();
		const textureView = context.getCurrentTexture().createView();

		const renderPassDescriptor: GPURenderPassDescriptor = {
			colorAttachments: [
				{
					view: textureView,
					clearValue: { r: 0.0, g: 0.0, b: 0.3, a: 1.0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		};

		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
		passEncoder.setPipeline(pipeline);
		passEncoder.draw(3, 1, 0, 0);
		passEncoder.end();
		const commandBuffer = commandEncoder.finish();
		device.queue.submit([commandBuffer]);
	};
	return update;
};

const setupPipeline = ({ context, device }: GPUResource) => {
	const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device,
		format: presentationFormat,
		alphaMode: "opaque",
	});

	const pipeline = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module: device.createShaderModule({
				code: triangleVertWGSL,
			}),
			entryPoint: "main",
		},
		fragment: {
			module: device.createShaderModule({
				code: fragWGSL,
			}),
			entryPoint: "main",
			targets: [
				{
					format: presentationFormat,
				},
			],
		},
		primitive: {
			topology: "triangle-list",
		},
	});
	return pipeline;
};

export const HelloWorld = () => {
	const resource = useGPUResource();
	const pipeline = useMemo(() => setupPipeline(resource), [resource]);
	const update = useMemo(
		() => createUpdater(resource, pipeline),
		[resource, pipeline],
	);

	useFrame(update, 30);

	return null;
};
