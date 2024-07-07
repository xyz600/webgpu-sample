import {
	type Context,
	type ReactNode,
	createContext,
	useCallback,
	useContext,
	useRef,
	useState,
} from "react";

export type GPUResource = {
	device: GPUDevice;
	context: GPUCanvasContext;
};

const getGPUResource = async (
	canvas: HTMLCanvasElement,
): Promise<GPUResource> => {
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
	const context = canvas.getContext("webgpu") as GPUCanvasContext;

	return { device, context };
};

const GPUResourceContext: Context<GPUResource> = createContext(
	{} as GPUResource,
);

export const useGPUResource = (): GPUResource => {
	return useContext(GPUResourceContext);
};

type InnerProps = {
	resource: GPUResource | undefined;
	children: ReactNode;
};

const WebGPUCanvasInner = ({ resource, children }: InnerProps) => {
	if (!resource) {
		return null;
	}

	return (
		<GPUResourceContext.Provider value={resource}>
			{children}
		</GPUResourceContext.Provider>
	);
};

export type Props = {
	id: string;
	children: ReactNode;
};

export const WebGPUCanvas = ({ id, children }: Props) => {
	const refCanvas = useRef<HTMLCanvasElement | null>();

	const [resource, setResource] = useState<GPUResource | undefined>();

	const callbackRef = useCallback(async (node: HTMLCanvasElement) => {
		refCanvas.current = node;
		const resource = await getGPUResource(node);
		if (resource) {
			setResource(resource);
		}
	}, []);

	return (
		<canvas id={id} height={800} width={1200} ref={callbackRef}>
			<WebGPUCanvasInner resource={resource}>{children}</WebGPUCanvasInner>
		</canvas>
	);
};
