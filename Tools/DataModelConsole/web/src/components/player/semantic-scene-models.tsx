"use client";

import { RoundedBox } from "@react-three/drei";
import type { ThreeElements } from "@react-three/fiber";
import { memo } from "react";
import { AdditiveBlending, DoubleSide } from "three";

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
} & ThreeElements["meshPhysicalMaterial"]) {
  return (
    <meshPhysicalMaterial
      color={color}
      clearcoat={0.92}
      clearcoatRoughness={0.16}
      emissive={color}
      emissiveIntensity={0.04 + confidence * 0.13}
      envMapIntensity={1.8}
      metalness={0.62}
      opacity={opacity}
      roughness={0.22}
      transparent={opacity < 1}
      {...props}
    />
  );
}

function GroundHalo({
  color,
  confidence,
  opacity,
  scale,
}: {
  color: string;
  confidence: number;
  opacity: number;
  scale: [number, number, number];
}) {
  return (
    <mesh
      position={[0, 0.035, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      scale={scale}
    >
      <ringGeometry args={[0.72, 1, 48]} />
      <meshBasicMaterial
        blending={AdditiveBlending}
        color={color}
        depthWrite={false}
        opacity={opacity * (0.16 + confidence * 0.28)}
        side={DoubleSide}
        toneMapped={false}
        transparent
      />
    </mesh>
  );
}

function Wheel({
  position,
  radius,
}: {
  position: ScenePosition;
  radius: number;
}) {
  return (
    <group position={position} rotation={[0, 0, Math.PI / 2]}>
      <mesh castShadow>
        <cylinderGeometry args={[radius, radius, radius * 0.58, 16]} />
        <meshPhysicalMaterial
          color="#05080a"
          envMapIntensity={1.4}
          metalness={0.45}
          roughness={0.5}
        />
      </mesh>
      <mesh position={[0, radius * 0.3, 0]}>
        <cylinderGeometry
          args={[radius * 0.54, radius * 0.54, radius * 0.04, 12]}
        />
        <meshPhysicalMaterial
          color="#a8b6bf"
          clearcoat={0.7}
          envMapIntensity={2}
          metalness={0.95}
          roughness={0.13}
        />
      </mesh>
    </group>
  );
}

export const EgoVehicle = memo(function EgoVehicle() {
  const wheelPositions: ScenePosition[] = [
    [-1.02, 0.4, -1.45],
    [1.02, 0.4, -1.45],
    [-1.02, 0.4, 1.46],
    [1.02, 0.4, 1.46],
  ];

  return (
    <group
      name="auto-e2e-ego-vehicle"
      position={[0, 0.02, 0]}
      scale={[1.08, 1.08, 1.08]}
    >
      <GroundHalo
        color="#5de9ff"
        confidence={1}
        opacity={0.95}
        scale={[1.48, 2.65, 1]}
      />
      <RoundedBox
        args={[2.05, 0.58, 4.65]}
        castShadow
        position={[0, 0.58, 0]}
        radius={0.24}
        receiveShadow
        smoothness={5}
      >
        <meshPhysicalMaterial
          clearcoat={1}
          clearcoatRoughness={0.08}
          color="#cbd7df"
          envMapIntensity={2.8}
          iridescence={0.2}
          iridescenceIOR={1.55}
          metalness={0.92}
          roughness={0.12}
        />
      </RoundedBox>
      <RoundedBox
        args={[1.62, 0.66, 2.42]}
        castShadow
        position={[0, 1.02, -0.18]}
        radius={0.3}
        smoothness={5}
      >
        <meshPhysicalMaterial
          clearcoat={1}
          clearcoatRoughness={0.04}
          color="#071217"
          envMapIntensity={3.2}
          metalness={0.7}
          roughness={0.06}
          thickness={0.35}
          transmission={0.16}
        />
      </RoundedBox>
      <RoundedBox
        args={[1.72, 0.16, 1.18]}
        position={[0, 0.83, 1.48]}
        radius={0.08}
        smoothness={3}
      >
        <meshPhysicalMaterial
          clearcoat={1}
          color="#e0e9ee"
          envMapIntensity={2.6}
          metalness={0.9}
          roughness={0.1}
        />
      </RoundedBox>
      {[-1, 1].map((side) => (
        <mesh
          key={`ego-side-${side}`}
          position={[side * 1.035, 0.56, 0.18]}
          scale={[0.025, 0.055, 1.58]}
        >
          <boxGeometry />
          <meshBasicMaterial
            color="#8ef5ff"
            toneMapped={false}
          />
        </mesh>
      ))}
      {[-0.58, 0.58].map((x) => (
        <RoundedBox
          key={`ego-headlight-${x}`}
          args={[0.58, 0.085, 0.06]}
          position={[x, 0.65, 2.34]}
          radius={0.03}
          smoothness={2}
        >
          <meshBasicMaterial
            color="#eaffff"
            toneMapped={false}
          />
        </RoundedBox>
      ))}
      <RoundedBox
        args={[1.42, 0.075, 0.055]}
        position={[0, 0.67, -2.34]}
        radius={0.025}
        smoothness={2}
      >
        <meshBasicMaterial color="#ff284f" toneMapped={false} />
      </RoundedBox>
      {wheelPositions.map((position) => (
        <Wheel key={position.join(":")} position={position} radius={0.38} />
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
  const radius = Math.min(0.28, width * 0.13);
  const wheelRadius = Math.min(0.38, width * 0.19);
  const axleX = width * 0.51;
  const axleZ = length * 0.31;
  const wheelPositions: ScenePosition[] = [
    [-axleX, wheelRadius, -axleZ],
    [axleX, wheelRadius, -axleZ],
    [-axleX, wheelRadius, axleZ],
    [axleX, wheelRadius, axleZ],
  ];

  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundHalo
        color={color}
        confidence={confidence}
        opacity={opacity}
        scale={[width * 0.72, length * 0.62, 1]}
      />
      <RoundedBox
        args={[width, 0.58, length]}
        castShadow
        position={[0, 0.57, 0]}
        radius={radius}
        receiveShadow
        smoothness={4}
      >
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
        />
      </RoundedBox>
      <RoundedBox
        args={[width * 0.76, 0.56, length * 0.5]}
        castShadow
        position={[0, 0.97, -length * 0.055]}
        radius={radius}
        smoothness={4}
      >
        <meshPhysicalMaterial
          clearcoat={0.96}
          clearcoatRoughness={0.06}
          color="#0b171d"
          envMapIntensity={2.5}
          metalness={0.62}
          opacity={opacity}
          roughness={0.08}
          transparent={opacity < 1}
        />
      </RoundedBox>
      <RoundedBox
        args={[width * 0.78, 0.12, length * 0.28]}
        position={[0, 0.82, length * 0.32]}
        radius={0.06}
        smoothness={3}
      >
        <SemanticMaterial
          color={color}
          confidence={confidence}
          opacity={opacity}
          metalness={0.76}
          roughness={0.16}
        />
      </RoundedBox>
      {wheelPositions.map((wheelPosition) => (
        <Wheel
          key={wheelPosition.join(":")}
          position={wheelPosition}
          radius={wheelRadius}
        />
      ))}
      {[-width * 0.27, width * 0.27].map((x) => (
        <mesh
          key={`headlight-${x}`}
          position={[x, 0.64, length * 0.505]}
          scale={[width * 0.17, 0.045, 0.025]}
        >
          <boxGeometry />
          <meshBasicMaterial color="#e8ffff" toneMapped={false} />
        </mesh>
      ))}
      <mesh
        position={[0, 0.64, -length * 0.505]}
        scale={[width * 0.31, 0.035, 0.025]}
      >
        <boxGeometry />
        <meshBasicMaterial
          color="#ff3359"
          opacity={opacity}
          toneMapped={false}
          transparent={opacity < 1}
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
      <GroundHalo
        color={color}
        confidence={confidence}
        opacity={opacity}
        scale={[0.55, 0.55, 1]}
      />
      <mesh castShadow position={[0, 1.72, 0]}>
        <icosahedronGeometry args={[0.22, 2]} />
        <meshPhysicalMaterial
          clearcoat={0.35}
          color="#d9b8a4"
          envMapIntensity={1.2}
          opacity={opacity}
          roughness={0.48}
          transparent={opacity < 1}
        />
      </mesh>
      <RoundedBox
        args={[0.52, 0.72, 0.34]}
        castShadow
        position={[0, 1.13, 0]}
        radius={0.15}
        smoothness={3}
      >
        <SemanticMaterial
          color={color}
          confidence={confidence}
          metalness={0.25}
          opacity={opacity}
          roughness={0.34}
        />
      </RoundedBox>
      {[-0.31, 0.31].map((x) => (
        <mesh
          key={`arm-${x}`}
          castShadow
          position={[x, 1.08, 0]}
          rotation={[0, 0, x < 0 ? -0.16 : 0.16]}
        >
          <capsuleGeometry args={[0.075, 0.52, 3, 7]} />
          <SemanticMaterial
            color={color}
            confidence={confidence}
            metalness={0.18}
            opacity={opacity}
            roughness={0.42}
          />
        </mesh>
      ))}
      {[-0.14, 0.14].map((x) => (
        <mesh
          key={`leg-${x}`}
          castShadow
          position={[x, 0.46, 0]}
          rotation={[0, 0, x < 0 ? -0.07 : 0.07]}
        >
          <capsuleGeometry args={[0.09, 0.55, 3, 7]} />
          <meshPhysicalMaterial
            color="#18252b"
            envMapIntensity={1.2}
            metalness={0.3}
            opacity={opacity}
            roughness={0.5}
            transparent={opacity < 1}
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
  const panelHeight = Math.max(0.55, height * 0.74);
  const cornerX = Math.max(0.2, width * 0.34);
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundHalo
        color={color}
        confidence={confidence}
        opacity={opacity}
        scale={[width * 0.68, length * 0.66, 1]}
      />
      <RoundedBox
        args={[width, panelHeight, length]}
        castShadow
        position={[0, panelHeight / 2, 0]}
        radius={Math.min(0.18, width * 0.12, length * 0.12)}
        receiveShadow
        smoothness={3}
      >
        <SemanticMaterial
          color={color}
          confidence={confidence}
          metalness={0.48}
          opacity={opacity}
          roughness={0.3}
        />
      </RoundedBox>
      <RoundedBox
        args={[width * 0.82, 0.12, length * 0.82]}
        position={[0, panelHeight + 0.05, 0]}
        radius={0.05}
        smoothness={2}
      >
        <meshPhysicalMaterial
          clearcoat={0.75}
          color="#202b30"
          envMapIntensity={1.7}
          metalness={0.75}
          opacity={opacity}
          roughness={0.22}
          transparent={opacity < 1}
        />
      </RoundedBox>
      {[-0.23, 0, 0.23].map((offset) => (
        <mesh
          key={`stripe-${offset}`}
          position={[
            offset * width,
            panelHeight * 0.54,
            length / 2 + 0.018,
          ]}
          rotation={[0, 0, -0.58]}
          scale={[width * 0.22, 0.045, 0.025]}
        >
          <boxGeometry />
          <meshBasicMaterial
            color={offset === 0 ? "#f8fbff" : color}
            opacity={opacity}
            toneMapped={false}
            transparent={opacity < 1}
          />
        </mesh>
      ))}
      {[-cornerX, cornerX].map((x) => (
        <mesh
          key={`beacon-${x}`}
          position={[x, panelHeight + 0.18, 0]}
        >
          <cylinderGeometry args={[0.09, 0.12, 0.25, 10]} />
          <meshBasicMaterial
            color={color}
            opacity={opacity}
            toneMapped={false}
            transparent={opacity < 1}
          />
        </mesh>
      ))}
    </group>
  );
});
