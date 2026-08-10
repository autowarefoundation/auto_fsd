"use client";

import { RoundedBox, useGLTF } from "@react-three/drei";
import type { ThreeElements } from "@react-three/fiber";
import {
  memo,
  Suspense,
  useEffect,
  useMemo,
} from "react";
import {
  AdditiveBlending,
  Color,
  DoubleSide,
  Material,
  Mesh,
  MeshStandardMaterial,
} from "three";

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

interface VehicleAsset {
  height: number;
  nativeSize: ScenePosition;
  path: string;
}

const ASSET_ROOT =
  "/assets/semantic-occupancy/kenney-car-kit";

const VEHICLE_ASSETS = {
  ego: {
    path: `${ASSET_ROOT}/race-future.glb`,
    nativeSize: [1.2, 0.833, 2.66],
    height: 1.14,
  },
  sedan: {
    path: `${ASSET_ROOT}/sedan-sports.glb`,
    nativeSize: [1.3, 1.1, 2.55],
    height: 1.5,
  },
  suv: {
    path: `${ASSET_ROOT}/suv-luxury.glb`,
    nativeSize: [1.5, 1.3, 2.85],
    height: 1.82,
  },
  van: {
    path: `${ASSET_ROOT}/van.glb`,
    nativeSize: [1.5, 1.35, 2.75],
    height: 2.05,
  },
} satisfies Record<string, VehicleAsset>;

const OBSTACLE_ASSETS = {
  box: {
    path: `${ASSET_ROOT}/box.glb`,
    nativeSize: [0.715, 0.715, 0.715] as ScenePosition,
  },
  cone: {
    path: `${ASSET_ROOT}/cone.glb`,
    nativeSize: [0.476, 0.595, 0.476] as ScenePosition,
  },
};

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

function SemanticAsset({
  color,
  confidence,
  opacity,
  path,
  scale,
  tintStrength,
}: {
  color: string;
  confidence: number;
  opacity: number;
  path: string;
  scale: ScenePosition;
  tintStrength: number;
}) {
  const { scene } = useGLTF(path, false, false);
  const clone = useMemo(() => {
    const next = scene.clone(true);
    const tint = new Color(color);
    next.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.castShadow = true;
      child.receiveShadow = true;
      const sourceMaterials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      const materials = sourceMaterials.map((source) => {
        const material = source.clone();
        if (material instanceof MeshStandardMaterial) {
          material.color.lerp(tint, tintStrength);
          material.emissive.copy(tint);
          material.emissiveIntensity =
            0.015 + confidence * 0.035;
          material.envMapIntensity = 2.1;
          material.opacity = opacity;
          material.roughness = Math.min(material.roughness, 0.34);
          material.transparent = opacity < 1;
          material.depthWrite = opacity >= 0.55;
        }
        return material;
      });
      child.material = Array.isArray(child.material)
        ? materials
        : materials[0];
    });
    return next;
  }, [color, confidence, opacity, scene, tintStrength]);

  useEffect(
    () => () => {
      clone.traverse((child) => {
        if (!(child instanceof Mesh)) return;
        const materials = Array.isArray(child.material)
          ? child.material
          : [child.material];
        materials.forEach((material: Material) => material.dispose());
      });
    },
    [clone],
  );

  return <primitive object={clone} scale={scale} />;
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
  scale: ScenePosition;
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

function VehicleFallback({
  color,
  length,
  opacity,
  width,
}: {
  color: string;
  length: number;
  opacity: number;
  width: number;
}) {
  return (
    <RoundedBox
      args={[width, Math.min(1, width * 0.48), length]}
      castShadow
      position={[0, Math.min(0.5, width * 0.24), 0]}
      radius={Math.min(0.25, width * 0.12)}
      smoothness={3}
    >
      <meshPhysicalMaterial
        clearcoat={0.8}
        color={color}
        metalness={0.55}
        opacity={opacity}
        roughness={0.24}
        transparent={opacity < 1}
      />
    </RoundedBox>
  );
}

export const EgoVehicle = memo(function EgoVehicle() {
  const asset = VEHICLE_ASSETS.ego;
  const width = 2.08;
  const length = 4.72;
  return (
    <group name="auto-e2e-ego-vehicle" position={[0, 0.025, 0]}>
      <GroundHalo
        color="#5de9ff"
        confidence={1}
        opacity={0.96}
        scale={[1.52, 2.7, 1]}
      />
      <Suspense
        fallback={
          <VehicleFallback
            color="#d7e3e9"
            length={length}
            opacity={1}
            width={width}
          />
        }
      >
        <SemanticAsset
          color="#5de9ff"
          confidence={1}
          opacity={1}
          path={asset.path}
          scale={[
            width / asset.nativeSize[0],
            asset.height / asset.nativeSize[1],
            length / asset.nativeSize[2],
          ]}
          tintStrength={0.08}
        />
      </Suspense>
      <mesh position={[0, 0.09, 0]} scale={[0.78, 0.025, 1.9]}>
        <boxGeometry />
        <meshBasicMaterial
          blending={AdditiveBlending}
          color="#42e7ff"
          depthWrite={false}
          opacity={0.2}
          toneMapped={false}
          transparent
        />
      </mesh>
    </group>
  );
});

function vehicleAssetFor(
  length: number,
  position: ScenePosition,
): VehicleAsset {
  if (length >= 7.2) return VEHICLE_ASSETS.van;
  if (position[0] < -3 || length >= 5.5) return VEHICLE_ASSETS.suv;
  return VEHICLE_ASSETS.sedan;
}

export const SemanticVehicle = memo(function SemanticVehicle({
  color,
  confidence,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticVehicleProps) {
  const asset = vehicleAssetFor(length, position);
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundHalo
        color={color}
        confidence={confidence}
        opacity={opacity}
        scale={[width * 0.72, length * 0.62, 1]}
      />
      <Suspense
        fallback={
          <VehicleFallback
            color={color}
            length={length}
            opacity={opacity}
            width={width}
          />
        }
      >
        <SemanticAsset
          color={color}
          confidence={confidence}
          opacity={opacity}
          path={asset.path}
          scale={[
            width / asset.nativeSize[0],
            asset.height / asset.nativeSize[1],
            length / asset.nativeSize[2],
          ]}
          tintStrength={0.2}
        />
      </Suspense>
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

function ObstacleFallback({
  color,
  height,
  length,
  opacity,
  width,
}: {
  color: string;
  height: number;
  length: number;
  opacity: number;
  width: number;
}) {
  return (
    <RoundedBox
      args={[width, height, length]}
      castShadow
      position={[0, height / 2, 0]}
      radius={Math.min(0.16, width * 0.1, length * 0.1)}
      smoothness={3}
    >
      <meshPhysicalMaterial
        clearcoat={0.6}
        color={color}
        metalness={0.4}
        opacity={opacity}
        roughness={0.32}
        transparent={opacity < 1}
      />
    </RoundedBox>
  );
}

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
  const isCone = width <= 1.5 && length <= 1.5;
  const asset = isCone
    ? OBSTACLE_ASSETS.cone
    : OBSTACLE_ASSETS.box;
  const targetHeight = isCone ? Math.max(0.7, height) : height;

  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundHalo
        color={color}
        confidence={confidence}
        opacity={opacity}
        scale={[width * 0.68, length * 0.66, 1]}
      />
      <Suspense
        fallback={
          <ObstacleFallback
            color={color}
            height={targetHeight}
            length={length}
            opacity={opacity}
            width={width}
          />
        }
      >
        <SemanticAsset
          color={color}
          confidence={confidence}
          opacity={opacity}
          path={asset.path}
          scale={[
            width / asset.nativeSize[0],
            targetHeight / asset.nativeSize[1],
            length / asset.nativeSize[2],
          ]}
          tintStrength={0.16}
        />
      </Suspense>
    </group>
  );
});
