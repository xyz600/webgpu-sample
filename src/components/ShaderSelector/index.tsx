import { Box, Link, Text, VStack } from "@chakra-ui/react";
import type { ShaderExampleType } from "./types";

type Props = {
	currentSelection: ShaderExampleType;
	onSelectionChanged: (type: ShaderExampleType) => void;
};

type ShaderExampleItem = {
	name: string;
	type: ShaderExampleType;
};

const shaderExampleList: ShaderExampleItem[] = [
	{
		name: "Hello World",
		type: "Hello World",
	},
	{
		name: "Canvas Histgram",
		type: "Canvas Histgram",
	},
	{
		name: "Matrix Multipulation",
		type: "MatMul",
	},
];

export const ShaderSelector = ({
	onSelectionChanged,
	currentSelection,
}: Props) => {
	const toLink = ({ name, type }: ShaderExampleItem) => {
		if (name === currentSelection) {
			return (
				<Text fontSize="lg" fontWeight="bold" key={name}>
					{name}
				</Text>
			);
		}
		return (
			<Link
				_hover={{ textDecoration: "none", bg: "gray.700" }}
				w="100%"
				p="2"
				key={name}
				onClick={() => {
					onSelectionChanged(type);
				}}
			>
				{name}
			</Link>
		);
	};

	return (
		<Box
			as="nav"
			pos="fixed"
			top="0"
			left="0"
			h="100%"
			w="200px"
			bg="gray.800"
			color="white"
			p="4"
		>
			<VStack align="start" spacing="4">
				{shaderExampleList.map((item) => toLink(item))}
			</VStack>
		</Box>
	);
};
