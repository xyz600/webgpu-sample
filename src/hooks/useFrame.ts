import { useEffect, useRef } from "react";

export const useFrame = (callback: () => void, fps: number) => {
	const refId = useRef<number | undefined>(undefined);

	useEffect(() => {
		const timeoutMs = 1_000 / fps;
		if (typeof refId.current === "number") {
			clearInterval(refId.current);
		}
		refId.current = undefined;
		const id = setInterval(() => callback(), timeoutMs);
		refId.current = id;
		return () => {
			if (typeof refId.current === "number") {
				clearInterval(id);
			}
		};
	}, [fps, callback]);
};
