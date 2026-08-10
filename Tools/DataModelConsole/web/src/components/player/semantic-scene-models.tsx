"use client";

import { Edges, RoundedBox, useGLTF } from "@react-three/drei";
import type { ThreeElements } from "@react-three/fiber";
import {
  memo,
  Suspense,
  useEffect,
  useMemo,
} from "react";
import {
  AdditiveBlending,
  BackSide,
  Color,
  Material,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from "three";

export {
  PremiumEgoVehicle as EgoVehicle,
} from "@/components/player/premium-ego-vehicle";

type ScenePosition = [number, number, number];

interface SemanticModelProps {
  color: string;
  confidence: number;
  errorKind?: "fp" | "fn" | null;
  opacity: number;
  position: ScenePosition;
}

interface SemanticVehicleProps extends SemanticModelProps {
  length: number;
  width: number;
  yaw: number;
}

interface SemanticPedestrianProps extends SemanticModelProps {
  length: number;
  width: number;
  yaw: number;
}

interface SemanticObstacleProps extends SemanticModelProps {
  decorativeVariants?: boolean;
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

type ObstacleKind =
  | "barrier"
  | "bicycle"
  | "bollard"
  | "cone"
  | "crate"
  | "drum"
  | "fence"
  | "guardrail"
  | "sign";

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

function SemanticAssetShell({
  color,
  confidence,
  errorKind,
  path,
  scale,
}: {
  color: string;
  confidence: number;
  errorKind?: "fp" | "fn" | null;
  path: string;
  scale: ScenePosition;
}) {
  const { scene } = useGLTF(path, false, false);
  const clone = useMemo(() => {
    const next = scene.clone(true);
    next.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.castShadow = false;
      child.receiveShadow = false;
      child.material = new MeshBasicMaterial({
        blending: AdditiveBlending,
        color,
        depthWrite: false,
        opacity: errorKind
          ? 0.14 + confidence * 0.12
          : 0.035 + confidence * 0.045,
        side: BackSide,
        toneMapped: false,
        transparent: true,
      });
    });
    return next;
  }, [color, confidence, errorKind, scene]);

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

  return (
    <primitive
      object={clone}
      scale={scale.map((value) => value * 1.025) as ScenePosition}
    />
  );
}

function GroundFootprint({
  color,
  confidence,
  length,
  opacity,
  width,
}: {
  color: string;
  confidence: number;
  length: number;
  opacity: number;
  width: number;
}) {
  return (
    <mesh position={[0, 0.03, 0]}>
      <boxGeometry args={[width, 0.018, length]} />
      <meshBasicMaterial
        color={color}
        depthWrite={false}
        opacity={opacity * (0.025 + confidence * 0.055)}
        toneMapped={false}
        transparent
      />
      <Edges
        color={color}
        threshold={1}
        toneMapped={false}
        transparent
        opacity={opacity * (0.26 + confidence * 0.42)}
      />
    </mesh>
  );
}

function ConfidencePillar({
  color,
  confidence,
  errorKind,
}: {
  color: string;
  confidence: number;
  errorKind?: "fp" | "fn" | null;
}) {
  const height = 0.12 + confidence * 0.72;

  return (
    <mesh position={[0, height / 2 + 0.04, 0]}>
      <cylinderGeometry args={[0.035, 0.065, height, 12]} />
      <meshBasicMaterial
        blending={AdditiveBlending}
        color={color}
        depthWrite={false}
        opacity={errorKind ? 0.7 : 0.18 + confidence * 0.32}
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
  errorKind,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticVehicleProps) {
  const asset = vehicleAssetFor(length, position);
  const assetScale: ScenePosition = [
    width / asset.nativeSize[0],
    asset.height / asset.nativeSize[1],
    length / asset.nativeSize[2],
  ];
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundFootprint
        color={color}
        confidence={confidence}
        length={length}
        opacity={opacity}
        width={width}
      />
      <ConfidencePillar
        color={color}
        confidence={confidence}
        errorKind={errorKind}
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
          scale={assetScale}
          tintStrength={0.2}
        />
        <SemanticAssetShell
          color={color}
          confidence={confidence}
          errorKind={errorKind}
          path={asset.path}
          scale={assetScale}
        />
      </Suspense>
    </group>
  );
});

export const SemanticPedestrian = memo(function SemanticPedestrian({
  color,
  confidence,
  errorKind,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticPedestrianProps) {
  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundFootprint
        color={color}
        confidence={confidence}
        length={length}
        opacity={opacity}
        width={width}
      />
      <ConfidencePillar
        color={color}
        confidence={confidence}
        errorKind={errorKind}
      />
      <group scale={[width / 0.82, 1, length / 0.82]}>
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

function Tube({
  color,
  from,
  opacity,
  radius = 0.035,
  to,
}: {
  color: string;
  from: ScenePosition;
  opacity: number;
  radius?: number;
  to: ScenePosition;
}) {
  const transform = useMemo(() => {
    const start = new Vector3(...from);
    const end = new Vector3(...to);
    const direction = end.clone().sub(start);
    return {
      length: direction.length(),
      midpoint: start.add(end).multiplyScalar(0.5),
      quaternion: new Quaternion().setFromUnitVectors(
        new Vector3(0, 1, 0),
        direction.normalize(),
      ),
    };
  }, [from, to]);

  return (
    <mesh
      position={transform.midpoint}
      quaternion={transform.quaternion}
    >
      <cylinderGeometry args={[radius, radius, transform.length, 8]} />
      <meshPhysicalMaterial
        clearcoat={0.65}
        color={color}
        envMapIntensity={1.7}
        metalness={0.72}
        opacity={opacity}
        roughness={0.24}
        transparent={opacity < 1}
      />
    </mesh>
  );
}

function BarrierModel({
  confidence,
  length,
  opacity,
}: {
  confidence: number;
  length: number;
  opacity: number;
}) {
  const segments = Math.max(3, Math.min(8, Math.round(length / 0.8)));
  return (
    <group>
      <RoundedBox
        args={[0.26, 0.34, length]}
        castShadow
        position={[0, 1.02, 0]}
        radius={0.08}
        smoothness={3}
      >
        <SemanticMaterial
          color="#f08b32"
          confidence={confidence}
          opacity={opacity}
        />
      </RoundedBox>
      {Array.from({ length: segments }, (_, index) => (
        <RoundedBox
          key={`barrier-stripe-${index}`}
          args={[0.28, 0.22, Math.max(0.18, length / segments / 2.2)]}
          position={[
            0,
            1.03,
            -length / 2 + ((index + 0.5) * length) / segments,
          ]}
          radius={0.025}
          smoothness={2}
        >
          <meshBasicMaterial color="#f7f4dc" toneMapped={false} />
        </RoundedBox>
      ))}
      {[-0.38, 0.38].map((ratio) => (
        <group key={`barrier-foot-${ratio}`} position={[0, 0, ratio * length]}>
          <RoundedBox
            args={[0.18, 0.86, 0.18]}
            castShadow
            position={[0, 0.52, 0]}
            radius={0.04}
            smoothness={2}
          >
            <meshPhysicalMaterial
              color="#202b31"
              metalness={0.68}
              opacity={opacity}
              roughness={0.3}
              transparent={opacity < 1}
            />
          </RoundedBox>
          <RoundedBox
            args={[0.82, 0.12, 0.36]}
            castShadow
            position={[0, 0.08, 0]}
            radius={0.05}
            smoothness={2}
          >
            <meshPhysicalMaterial
              color="#11191d"
              metalness={0.52}
              opacity={opacity}
              roughness={0.38}
              transparent={opacity < 1}
            />
          </RoundedBox>
        </group>
      ))}
    </group>
  );
}

function BollardModel({
  confidence,
  opacity,
}: {
  confidence: number;
  opacity: number;
}) {
  return (
    <group>
      <mesh castShadow position={[0, 0.55, 0]}>
        <cylinderGeometry args={[0.15, 0.2, 1.1, 16]} />
        <SemanticMaterial
          color="#e76738"
          confidence={confidence}
          opacity={opacity}
        />
      </mesh>
      <mesh position={[0, 0.75, 0]}>
        <cylinderGeometry args={[0.156, 0.17, 0.18, 16]} />
        <meshBasicMaterial color="#f2f6eb" toneMapped={false} />
      </mesh>
      <mesh castShadow position={[0, 0.06, 0]}>
        <cylinderGeometry args={[0.34, 0.4, 0.12, 16]} />
        <meshPhysicalMaterial
          color="#172127"
          metalness={0.74}
          roughness={0.3}
        />
      </mesh>
    </group>
  );
}

function RoadSignModel({
  confidence,
  opacity,
}: {
  confidence: number;
  opacity: number;
}) {
  return (
    <group>
      <mesh castShadow position={[0, 0.92, 0]}>
        <cylinderGeometry args={[0.045, 0.055, 1.84, 12]} />
        <meshPhysicalMaterial
          color="#9aa7ad"
          envMapIntensity={1.8}
          metalness={0.92}
          roughness={0.22}
        />
      </mesh>
      <mesh
        castShadow
        position={[0, 1.78, 0]}
        rotation={[Math.PI / 2, 0, 0]}
      >
        <cylinderGeometry args={[0.48, 0.48, 0.075, 8]} />
        <SemanticMaterial
          color="#e64c58"
          confidence={confidence}
          opacity={opacity}
        />
      </mesh>
      <mesh position={[0, 1.78, -0.042]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 0.39, 8]} />
        <meshBasicMaterial color="#f8f8ec" toneMapped={false} />
      </mesh>
      <mesh castShadow position={[0, 0.06, 0]}>
        <cylinderGeometry args={[0.28, 0.34, 0.12, 16]} />
        <meshPhysicalMaterial
          color="#182329"
          metalness={0.68}
          roughness={0.32}
        />
      </mesh>
    </group>
  );
}

function BicycleModel({
  color,
  length,
  opacity,
}: {
  color: string;
  length: number;
  opacity: number;
}) {
  const wheelOffset = Math.min(0.84, Math.max(0.62, length * 0.34));
  const rear: ScenePosition = [0, 0.5, -wheelOffset];
  const front: ScenePosition = [0, 0.5, wheelOffset];
  const crank: ScenePosition = [0, 0.54, -0.06];
  const seat: ScenePosition = [0, 1.15, -0.34];
  const handle: ScenePosition = [0, 1.12, wheelOffset * 0.72];

  return (
    <group>
      {[-wheelOffset, wheelOffset].map((z) => (
        <group key={`bicycle-wheel-${z}`} position={[0, 0.5, z]}>
          <mesh castShadow rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[0.46, 0.045, 10, 28]} />
            <meshPhysicalMaterial
              color="#101719"
              metalness={0.35}
              opacity={opacity}
              roughness={0.44}
              transparent={opacity < 1}
            />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <circleGeometry args={[0.055, 16]} />
            <meshPhysicalMaterial
              color="#b9c3c6"
              metalness={0.9}
              roughness={0.2}
            />
          </mesh>
        </group>
      ))}
      <Tube color={color} from={rear} opacity={opacity} to={crank} />
      <Tube color={color} from={crank} opacity={opacity} to={front} />
      <Tube color={color} from={crank} opacity={opacity} to={seat} />
      <Tube color={color} from={seat} opacity={opacity} to={rear} />
      <Tube color={color} from={seat} opacity={opacity} to={handle} />
      <Tube color="#c5d1d4" from={front} opacity={opacity} to={handle} />
      <Tube
        color="#c5d1d4"
        from={[-0.28, 1.16, handle[2]]}
        opacity={opacity}
        radius={0.025}
        to={[0.28, 1.16, handle[2]]}
      />
      <RoundedBox
        args={[0.22, 0.07, 0.32]}
        position={[0, 1.21, -0.39]}
        radius={0.03}
        smoothness={2}
      >
        <meshPhysicalMaterial
          color="#151b1d"
          metalness={0.38}
          roughness={0.42}
        />
      </RoundedBox>
    </group>
  );
}

function GuardrailModel({
  length,
  opacity,
  width,
}: {
  length: number;
  opacity: number;
  width: number;
}) {
  const postCount = Math.max(2, Math.min(9, Math.ceil(length / 1.6)));
  return (
    <group>
      <RoundedBox
        args={[Math.min(width, 0.18), 0.2, length]}
        castShadow
        position={[0, 0.68, 0]}
        radius={0.04}
        smoothness={2}
      >
        <meshPhysicalMaterial
          color="#a7b1b5"
          envMapIntensity={2}
          metalness={0.9}
          opacity={opacity}
          roughness={0.2}
          transparent={opacity < 1}
        />
      </RoundedBox>
      {Array.from({ length: postCount }, (_, index) => (
        <RoundedBox
          key={`guardrail-post-${index}`}
          args={[Math.min(width, 0.14), 0.72, 0.14]}
          castShadow
          position={[
            0,
            0.36,
            -length / 2 + ((index + 0.5) * length) / postCount,
          ]}
          radius={0.025}
          smoothness={2}
        >
          <meshPhysicalMaterial
            color="#77848a"
            metalness={0.84}
            opacity={opacity}
            roughness={0.26}
            transparent={opacity < 1}
          />
        </RoundedBox>
      ))}
    </group>
  );
}

function TrafficDrumModel({
  confidence,
  length,
  opacity,
  width,
}: {
  confidence: number;
  length: number;
  opacity: number;
  width: number;
}) {
  return (
    <group scale={[width / 0.72, 1, length / 0.72]}>
      <mesh castShadow position={[0, 0.46, 0]}>
        <cylinderGeometry args={[0.24, 0.34, 0.82, 16]} />
        <SemanticMaterial
          color="#ef6e2f"
          confidence={confidence}
          opacity={opacity}
        />
      </mesh>
      {[0.29, 0.55].map((y) => (
        <mesh key={`drum-stripe-${y}`} position={[0, y, 0]}>
          <cylinderGeometry args={[0.275, 0.29, 0.12, 16]} />
          <meshBasicMaterial color="#f2f4e9" toneMapped={false} />
        </mesh>
      ))}
      <RoundedBox
        args={[0.7, 0.1, 0.7]}
        castShadow
        position={[0, 0.05, 0]}
        radius={0.05}
        smoothness={2}
      >
        <meshPhysicalMaterial
          color="#11191d"
          metalness={0.4}
          roughness={0.46}
        />
      </RoundedBox>
    </group>
  );
}

function ConstructionFenceModel({
  length,
  opacity,
  width,
}: {
  length: number;
  opacity: number;
  width: number;
}) {
  const panelCount = Math.max(2, Math.min(10, Math.ceil(length / 1.2)));
  return (
    <group>
      {[-0.68, 0.68].map((y) => (
        <RoundedBox
          key={`fence-rail-${y}`}
          args={[Math.min(width, 0.08), 0.07, length]}
          position={[0, 0.86 + y * 0.62, 0]}
          radius={0.02}
          smoothness={2}
        >
          <meshPhysicalMaterial
            color="#b7c2c6"
            metalness={0.88}
            opacity={opacity}
            roughness={0.22}
            transparent={opacity < 1}
          />
        </RoundedBox>
      ))}
      {Array.from({ length: panelCount + 1 }, (_, index) => (
        <RoundedBox
          key={`fence-post-${index}`}
          args={[Math.min(width, 0.08), 1.45, 0.08]}
          castShadow
          position={[
            0,
            0.78,
            -length / 2 + (index * length) / panelCount,
          ]}
          radius={0.02}
          smoothness={2}
        >
          <meshPhysicalMaterial
            color="#8d9ba0"
            metalness={0.82}
            opacity={opacity}
            roughness={0.28}
            transparent={opacity < 1}
          />
        </RoundedBox>
      ))}
      {Array.from({ length: panelCount }, (_, index) => (
        <mesh
          key={`fence-panel-${index}`}
          position={[
            0,
            0.79,
            -length / 2 + ((index + 0.5) * length) / panelCount,
          ]}
        >
          <boxGeometry
            args={[
              Math.min(width, 0.035),
              1.1,
              Math.max(0.08, length / panelCount - 0.08),
            ]}
          />
          <meshBasicMaterial
            color="#d8e1e3"
            opacity={opacity * 0.12}
            transparent
            wireframe
          />
        </mesh>
      ))}
    </group>
  );
}

function obstacleKindFor({
  decorativeVariants,
  length,
  position,
  width,
}: Pick<
  SemanticObstacleProps,
  "decorativeVariants" | "length" | "position" | "width"
>): ObstacleKind {
  if (!decorativeVariants) return "crate";
  const major = Math.max(width, length);
  const minor = Math.min(width, length);
  const seed = Math.abs(
    Math.round(position[0] * 7 + position[2] * 11),
  );
  if (major >= 5) return seed % 2 === 0 ? "guardrail" : "fence";
  if (major >= 3.8) return "barrier";
  if (major >= 2 && minor <= 1.8) {
    return ["bicycle", "sign", "fence"][seed % 3] as ObstacleKind;
  }
  if (major <= 1.45) {
    return ["cone", "bollard", "drum"][seed % 3] as ObstacleKind;
  }
  return "crate";
}

function obstacleDisplayHeight(kind: ObstacleKind): number {
  switch (kind) {
    case "sign":
      return 2.25;
    case "fence":
      return 1.5;
    case "bicycle":
      return 1.28;
    case "barrier":
      return 1.2;
    case "bollard":
      return 1.1;
    case "crate":
      return 1.05;
    case "drum":
      return 0.9;
    case "guardrail":
      return 0.8;
    case "cone":
      return 0.75;
  }
}

export const SemanticObstacle = memo(function SemanticObstacle({
  color,
  confidence,
  decorativeVariants = false,
  errorKind,
  length,
  opacity,
  position,
  width,
  yaw,
}: SemanticObstacleProps) {
  const kind = obstacleKindFor({
    decorativeVariants,
    length,
    position,
    width,
  });
  const asset =
    kind === "cone" ? OBSTACLE_ASSETS.cone : OBSTACLE_ASSETS.box;
  const targetHeight = obstacleDisplayHeight(kind);

  return (
    <group position={position} rotation={[0, yaw, 0]}>
      <GroundFootprint
        color={color}
        confidence={confidence}
        length={length}
        opacity={opacity}
        width={width}
      />
      <ConfidencePillar
        color={color}
        confidence={confidence}
        errorKind={errorKind}
      />
      {kind === "barrier" ? (
        <BarrierModel
          confidence={confidence}
          length={length}
          opacity={opacity}
        />
      ) : kind === "bollard" ? (
        <BollardModel confidence={confidence} opacity={opacity} />
      ) : kind === "drum" ? (
        <TrafficDrumModel
          confidence={confidence}
          length={length}
          opacity={opacity}
          width={width}
        />
      ) : kind === "fence" ? (
        <ConstructionFenceModel
          length={length}
          opacity={opacity}
          width={width}
        />
      ) : kind === "guardrail" ? (
        <GuardrailModel
          length={length}
          opacity={opacity}
          width={width}
        />
      ) : kind === "sign" ? (
        <RoadSignModel confidence={confidence} opacity={opacity} />
      ) : kind === "bicycle" ? (
        <BicycleModel color={color} length={length} opacity={opacity} />
      ) : (
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
      )}
      {kind === "crate" && (
        <mesh position={[0, targetHeight + 0.04, 0]}>
          <boxGeometry args={[width * 0.72, 0.025, length * 0.72]} />
          <meshBasicMaterial
            blending={AdditiveBlending}
            color={color}
            depthWrite={false}
            opacity={0.16 + confidence * 0.14}
            toneMapped={false}
            transparent
          />
        </mesh>
      )}
    </group>
  );
});
