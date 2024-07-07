import { HStack, Heading, VStack } from "@chakra-ui/react";
import { useState } from "react";
import { ShaderSelector } from "./components/ShaderSelector";
import type { ShaderExampleType } from "./components/ShaderSelector/types";
import { CanvasHistgram } from "./components/shader/canvas-histgram";
import { HelloWorld } from "./components/shader/hello-world";
import { MatrixMultipulation } from "./components/shader/matrix-multiply";

const CanvasExample = ({ type }: { type: ShaderExampleType }) => {
	if (type === "Hello World") {
		return <HelloWorld />;
	}
	if (type === "Canvas Histgram") {
		return <CanvasHistgram />;
	}
	if (type === "MatMul") {
		return <MatrixMultipulation />;
	}
	return null;
};

function App() {
	const [shaderType, setShaderType] =
		useState<ShaderExampleType>("Hello World");

	return (
		<HStack h="100vw" height="100vh">
			<ShaderSelector
				currentSelection={shaderType}
				onSelectionChanged={setShaderType}
			/>
			<VStack height="100%" width="100%">
				<Heading>WebGPU Example</Heading>
				<CanvasExample type={shaderType} />
			</VStack>
		</HStack>
	);
}

export default App;
