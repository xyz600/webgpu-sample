import { WebGPUCanvas } from "./components/WebGPUCanvas";
import { HelloWorld } from "./components/shader/hello-world";

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
