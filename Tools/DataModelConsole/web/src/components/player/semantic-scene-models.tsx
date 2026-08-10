"use client";

import type { ThreeElements } from "@react-three/fiber";
import { memo } from "react";

type ScenePosition = [number, number, number];

interface SemanticModelProps {
  color: string;
  confidence: number;
  opacity: number;
  position: ScenePosition;
}

interface SemanticVehicleProps extends SemanticModelProps {
  length: number;
  width: number;
  yaw: number;
}

interface SemanticObstacleProps extends SemanticModelProps {
  height: number;
  length: number;
  width: number;
  yaw: number;
}

function SemanticMaterial({
  color,
  confidence,
  opacity,
  ...props
}: {
  color: string;
  confidence: number;
  opacity: number;
} & ThreeElements["meshStandardMaterial"]) {
  return (
    <meshStandardMaterial
      color={color}
      emissive={color}
      emissiveIntensity={0.08 + confidence * 0.22}
      metalness={0.28}
      roughness={0.42}
      transparent={opacity < 1}
      opacity={opacity}
      {...props}
    />
  );
}

export const EgoVehicle = memo(function EgoVehicle() {
  const wheelPositions: ScenePosition[] = [
    [-0.92, 0.42, -1.28],
    [0.92, 0.42, -1.28],
    [-0.92, 0.42, 1.3],
    [0.92, 0.42, 1.3],
  ];

  return (
    <group position={[0, 0.03, 0]} name="auto-e2e-ego-vehicle">
      <mesh
        castShadow
        receiveShadow
        position={[0, 0.58, 0]}
        rotation={[Math.PI / 2, 0, 0]}
        scale={[1, 1, 0.42]}
      >
        <capsuleGeometry args={[0.92, 2.7, 5, 16]} />
        <meshPhysicalMaterial
          color="#d7e1e8"
          clearcoat={1}
          clearcoatRoughness={0.14}
          metalness={0.78}
          roughness={0.2}
        />
      </mesh>
      <mesh
        castShadow
        position={[0, 0.9, -0.12]}
        rotation={[Math.PI / 2, 0, 0]}
        scale={[0.78, 1, 0.46]}
      >
        <capsuleGeometry args={[0.72, 1.55, 5, 12]} />
        <meshPhysicalMaterial
          color="#121b21"
          clearcoat={0.9}
          metalness={0.55}
          roughness={0.16}
        />
      </mesh>
      <mesh position={[0, 1.07, 0.15]} scale={[0.06, 0.025, 1.25]}>
        <boxGeometry />
        <meshStandardMaterial
          color="#8ef3ff"
          emissive="#36d9ee"
          emissiveIntensity={3}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[0, 0.61, 1.76]} scale={[0.64, 0.055, 0.035]}>
        <boxGeometry />
        <meshStandardMaterial
          color="#dffcff"
          emissive="#7ceeff"
          emissiveIntensity={5}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[0, 0.59, -1.78]} scale={[0.66, 0.045, 0.035]}>
        <boxGeometry />
        <meshStandardMaterial
          color="#ff385b"
          emissive="#ff163f"
          emissiveIntensity={4}
          toneMapped={false}
        />
      </mesh>
      {wheelPositions.map((position) => (
        <mesh
          key={position.join(":")}
          castShadow
          position={position}
          rotation={[0, 0, Math.PI / 2]}
        >
          <cylinderGeometry args={[0.36, 0.36, 0.22, 12]} />
          <meshStandardMaterial
            color="#070a0c"
            metalness={0.5}
            roughness={0.7}
          />
        </mesh>
      ))}
    </group>
  );
});

export const SemanticVehicle = memo(function SemanticVehicle({
  color,
  confidence,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticVehicleProps) {
  const bodyLength = Math.max(1.4, length - width);
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <mesh castShadow receiveShadow position={[0, 0.55, 0]}>
        <boxGeometry args={[width, 0.58, length]} />
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
        />
      </mesh>
      <mesh
        castShadow
        position={[0, 0.86, -length * 0.04]}
        rotation={[Math.PI / 2, 0, 0]}
        scale={[0.76, 0.75, 0.5]}
      >
        <capsuleGeometry
          args={[width * 0.5, bodyLength, 3, 8]}
        />
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
          metalness={0.45}
          roughness={0.28}
        />
      </mesh>
      <mesh
        position={[0, 0.68, length * 0.51]}
        scale={[width * 0.3, 0.035, 0.035]}
      >
        <boxGeometry />
        <meshStandardMaterial
          color="#d8fbff"
          emissive={color}
          emissiveIntensity={2.2}
          toneMapped={false}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
    </group>
  );
});

export const SemanticPedestrian = memo(function SemanticPedestrian({
  color,
  confidence,
  opacity,
  position,
}: SemanticModelProps) {
  return (
    <group position={position}>
      <mesh castShadow position={[0, 1.72, 0]}>
        <icosahedronGeometry args={[0.22, 1]} />
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
          metalness={0.08}
          roughness={0.55}
        />
      </mesh>
      <mesh castShadow position={[0, 1.06, 0]}>
        <capsuleGeometry args={[0.26, 0.72, 3, 8]} />
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
          metalness={0.08}
          roughness={0.55}
        />
      </mesh>
      {[-0.13, 0.13].map((x) => (
        <mesh
          key={x}
          castShadow
          position={[x, 0.42, 0]}
          rotation={[0, 0, x < 0 ? -0.08 : 0.08]}
        >
          <capsuleGeometry args={[0.08, 0.52, 3, 6]} />
          <SemanticMaterial
            color={color}
            confidence={confidence}
            opacity={opacity}
            metalness={0.08}
            roughness={0.6}
          />
        </mesh>
      ))}
    </group>
  );
});

export const SemanticObstacle = memo(function SemanticObstacle({
  color,
  confidence,
  height,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticObstacleProps) {
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <mesh castShadow receiveShadow position={[0, height / 2, 0]}>
        <boxGeometry args={[width, height, length]} />
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
          metalness={0.18}
          roughness={0.62}
        />
      </mesh>
      <mesh
        position={[0, height + 0.08, 0]}
        rotation={[0, Math.PI / 4, 0]}
      >
        <octahedronGeometry
          args={[Math.min(width, length, 1.2) * 0.35, 0]}
        />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.7 + confidence}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
    </group>
  );
});
