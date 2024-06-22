import { HelloWorld } from "./HelloWorld";
import { WebGPUCanvas } from "./WebGPUCanvas";

function App() {
	return (
		<>
			<h1>WebGPU Example</h1>
			<WebGPUCanvas id="webgpu-canvas">
				<HelloWorld />
			</WebGPUCanvas>
		</>
	);
}

export default App;
