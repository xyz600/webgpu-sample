import { useCallback, useRef, useState } from "react";
import { useFrame } from "../../../hooks/useFrame";
import fragWGSL from "./shaders/red.flag.wgsl";
import triangleVertWGSL from "./shaders/triangle.vert.wgsl";
import { Button, VStack } from "@chakra-ui/react";
import { getGPUDevice } from "../../../utils/device";

const createUpdater = (
	device: GPUDevice,
	context: GPUCanvasContext,
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

const setupPipeline = (device: GPUDevice, context: GPUCanvasContext) => {
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

const HelloWorldInner = ({
	gpuContext,
}: { gpuContext: GPUCanvasContext | undefined }) => {
	const [update, setUpdate] = useState<() => void>(() => {
		return () => {};
	});

	const triggerRender = async () => {
		if (!gpuContext) {
			return;
		}
		const device = await getGPUDevice();
		const pipeline = setupPipeline(device, gpuContext);
		const update = createUpdater(device, gpuContext, pipeline);

		setUpdate(() => {
			return update;
		});
	};

	useFrame(update, 30);

	return (
		<VStack>
			<Button onClick={triggerRender}>描画</Button>
		</VStack>
	);
};

export const HelloWorld = () => {
	const refCanvas = useRef<HTMLCanvasElement | null>();
	const [context, setContext] = useState<GPUCanvasContext>();
	const callbackRef = useCallback(async (node: HTMLCanvasElement | null) => {
		refCanvas.current = node;
		if (!node) {
			setContext(undefined);
			return;
		}
		const context = node.getContext("webgpu");
		if (context) {
			setContext(context as GPUCanvasContext);
		}
	}, []);

	return (
		<VStack>
			<canvas height="800px" width="1200px" ref={callbackRef} />
			<HelloWorldInner gpuContext={context} />
		</VStack>
	);
};
