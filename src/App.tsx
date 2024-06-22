import { HStack, Heading, VStack } from "@chakra-ui/react";
import { useState } from "react";
import { ShaderSelector } from "./components/ShaderSelector";
import type { ShaderExampleType } from "./components/ShaderSelector/types";
import { WebGPUCanvas } from "./components/WebGPUCanvas";
import { HelloWorld } from "./components/shader/hello-world";

function App() {
	const [shaderType, setShaderType] =
		useState<ShaderExampleType>("Hello World");

	return (
		<VStack>
			<Heading>WebGPU Example</Heading>
			<HStack>
				<ShaderSelector
					currentSelection={shaderType}
					onSelectionChanged={setShaderType}
				/>
				<WebGPUCanvas id="webgpu-canvas">
					<HelloWorld />
				</WebGPUCanvas>
			</HStack>
		</VStack>
	);
}

export default App;
